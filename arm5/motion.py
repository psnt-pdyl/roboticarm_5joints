"""Motion PINN for the 5-joint arm: joint angles over time for a rest-to-rest move.

    q(s) = q0 + (q1 - q0) b(s) + w(s) N(q0, q1 - q0, s, T, payload),   s = t / T in [0, 1]

b(s) = 10s^3 - 15s^4 + 6s^5 and w(s) = 64 s^3 (1 - s)^3 make the start and end exact (right
poses, zero speed and acceleration); the network N only shapes the path. Also: the page's
joint blend for this arm, as the baseline.
"""
import math

import torch
from torch import nn

from kin import LIM, N, Q_HALF, Q_MID

D = math.pi / 180
VMAX = torch.tensor([110.0, 130.0, 160.0, 200.0, 240.0]) * D     # the page's joint speed limits at 100 %
AMAX = torch.tensor([320.0, 380.0, 480.0, 600.0, 700.0]) * D     # and acceleration limits
T_REF, T_MIN = 1.0, 0.15


def basis(s):
    b = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    bd = 30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4
    bdd = 60 * s - 180 * s ** 2 + 120 * s ** 3
    w = 64 * s ** 3 * (1 - s) ** 3
    wd = 64 * (3 * s ** 2 * (1 - s) ** 3 - 3 * s ** 3 * (1 - s) ** 2)
    wdd = 64 * (6 * s * (1 - s) ** 3 - 18 * s ** 2 * (1 - s) ** 2 + 6 * s ** 3 * (1 - s))
    return (b, bd, bdd), (w, wd, wdd)


def compose(shape_fn, q0, q1, s, T):
    """q, qd, qdd in physical time for q(s) = q0 + (q1 - q0) b(s) + w(s) shape_fn(s)."""
    s = s.detach().requires_grad_(True)
    Nn = shape_fn(s)
    dN = torch.stack([torch.autograd.grad(Nn[..., j].sum(), s, create_graph=True)[0] for j in range(N)], -1)
    ddN = torch.stack([torch.autograd.grad(dN[..., j].sum(), s, create_graph=True)[0] for j in range(N)], -1)
    (b, bd, bdd), (w, wd, wdd) = basis(s)
    dq = q1 - q0
    q = q0 + dq * b[..., None] + w[..., None] * Nn
    q_s = dq * bd[..., None] + wd[..., None] * Nn + w[..., None] * dN
    q_ss = dq * bdd[..., None] + wdd[..., None] * Nn + 2 * wd[..., None] * dN + w[..., None] * ddN
    Tt = T[..., None]
    return q, q_s / Tt, q_ss / Tt ** 2


class MotionPINN5(nn.Module):
    def __init__(self, hidden=256, layers=4):
        super().__init__()
        dims = [2 * N + 3] + [hidden] * layers
        self.layers = nn.ModuleList(nn.Linear(i, o) for i, o in zip(dims, dims[1:]))
        self.out = nn.Linear(hidden, N)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("q_mid", Q_MID.clone())
        self.register_buffer("q_half", Q_HALF.clone())

    def shape(self, q0, q1, s, T, payload):
        x = torch.cat([(q0 - self.q_mid) / self.q_half, (q1 - q0) / self.q_half, (2 * s - 1)[..., None],
                       torch.log(T / T_REF)[..., None], (payload / 12.0)[..., None]], -1)
        for layer in self.layers:
            x = nn.functional.silu(layer(x))
        return self.out(x)

    def forward(self, q0, q1, s, T, payload):
        (b, _, _), (w, _, _) = basis(s)
        return q0 + (q1 - q0) * b[..., None] + w[..., None] * self.shape(q0, q1, s, T, payload)

    def motion(self, q0, q1, s, T, payload):
        return compose(lambda s_: self.shape(q0, q1, s_, T, payload), q0, q1, s, T)


def move_time(q0, q1, override=1.0):
    """The page's joint-move time: each joint's trapezoid, the slowest one wins."""
    o = torch.as_tensor(override, dtype=q0.dtype, device=q0.device)
    o = o[..., None] if o.dim() else o
    v, a = VMAX.to(q0.device) * o, AMAX.to(q0.device) * o
    d = (q1 - q0).abs()
    return torch.where(d >= v * v / a, d / v + v / a, 2 * torch.sqrt(d / a)).max(-1).values


def blend(q0, q1, t, override=1.0):
    """The page's joint blend: q, qd, qdd [..., k, N] at times t [..., k]."""
    o = torch.as_tensor(override, dtype=q0.dtype, device=q0.device)
    o = o[..., None, None] if o.dim() else o
    v, acc = VMAX.to(q0.device) * o, AMAX.to(q0.device) * o
    dlt = (q1 - q0)[..., None, :]
    d, sgn = dlt.abs(), torch.sign(dlt)
    cruise = d >= v * v / acc
    ta = torch.where(cruise, v / acc, torch.sqrt(d / acc))
    vp, tt = acc * ta, torch.where(cruise, d / v + v / acc, 2 * ta)
    t = torch.minimum(t[..., None].clamp(min=0), tt)
    tc = tt - ta
    s = torch.where(t < ta, 0.5 * acc * t ** 2, torch.where(t < tc, 0.5 * acc * ta ** 2 + vp * (t - ta), d - 0.5 * acc * (tt - t) ** 2))
    sd = torch.where(t < ta, acc * t, torch.where(t < tc, vp, acc * (tt - t)))
    sdd = torch.where(t >= tt, torch.zeros_like(t), torch.where(t < ta, acc, torch.where(t < tc, torch.zeros_like(t), -acc)))
    return q0[..., None, :] + sgn * s, sgn * sd, sgn * sdd
