"""One graph network for the 5-joint arm's whole move: joint angles at any moment, trained on the physics.

Same graph as model.ChainGNN (5 joints, the tool point, the goal), with three more inputs: each joint node gets its
start angle, and every node (time_all; only the goal node in the first version) gets the moment asked about and the
move's duration. Each joint decodes an angle e
(kept inside its limits by tanh), and the move is

    q(s) = q0 + b(s) * (e(b(s)) - q0),     s = t / T in [0, 1],   b(s) = 10s^3 - 15s^4 + 6s^5

so it starts exactly at q0, starts and stops at rest (b' = b'' = 0 at both ends), ends at e(1), the network's end
pose for the target, and never leaves the joint limits (q is between q0 and e). Trained on the pose GNN's loss at the
end of the move and the motion PINN's loss along it (train_mgnn.py): one network instead of the pose GNN + PINN pair.
"""
import math

import torch
from torch import nn

from kin import N
from model import EDGE_IN, GOAL, MLP, N_NODES, NODE_IN, ChainGNN
from motion import T_REF

EXTRA = 3                         # start angle (joint nodes), moment and log duration (goal node)
FD = 0.005                        # step in s for the speed and acceleration by central differences


def blend_b(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


class MotionGNN(nn.Module):
    def __init__(self, hidden=128, steps=6, time_all=True):
        super().__init__()
        base = ChainGNN(hidden, steps)
        self.hidden, self.steps, self.time_all = hidden, steps, time_all   # time_all: moment and duration on every node
        for name in ("node0", "edge0", "src", "dst", "q_mid", "q_half"):
            self.register_buffer(name, getattr(base, name).clone())
        self.enc_node = MLP(NODE_IN + EXTRA, hidden, hidden)
        self.enc_edge = MLP(EDGE_IN, hidden, hidden)
        self.edge_mlps = nn.ModuleList(MLP(3 * hidden, hidden, hidden) for _ in range(steps))
        self.node_mlps = nn.ModuleList(MLP(2 * hidden, hidden, hidden) for _ in range(steps))
        self.dec = MLP(hidden, hidden, 1, norm=False)

    @classmethod
    def from_pose_gnn(cls, state, hidden=128, steps=6, time_all=True):
        """Start from a trained pose GNN: the new inputs get zero weights, so it begins as a plain smooth move to the
        pose GNN's answer, whatever the start pose, duration or moment."""
        m = cls(hidden, steps, time_all)
        own = m.state_dict()
        for k, v in state.items():
            if k == "enc_node.l1.weight":
                own[k].zero_()
                own[k][:, :NODE_IN] = v
            else:
                own[k] = v
        m.load_state_dict(own)
        return m

    def end_angles(self, q0, goal, sigma, T):
        """e [..., N]: q0 [..., N], goal features [..., 5], sigma [...] (= b(s)), T [...] in s."""
        shape = q0.shape[:-1]
        q0, goal = q0.reshape(-1, N), goal.reshape(-1, goal.shape[-1])
        sigma, T = sigma.reshape(-1), T.reshape(-1)
        B = q0.shape[0]
        x = torch.zeros(B, N_NODES, NODE_IN + EXTRA, device=q0.device, dtype=q0.dtype)
        x[:, :, :NODE_IN] = self.node0
        x[:, GOAL, N_NODES:NODE_IN] = goal
        x[:, :N, NODE_IN] = (q0 - self.q_mid) / self.q_half
        nodes = slice(None) if self.time_all else GOAL
        x[:, nodes, NODE_IN + 1] = (2 * sigma - 1)[:, None] if self.time_all else 2 * sigma - 1
        x[:, nodes, NODE_IN + 2] = torch.log(T / T_REF)[:, None] if self.time_all else torch.log(T / T_REF)
        h = self.enc_node(x)
        e = self.enc_edge(self.edge0).expand(B, -1, -1)
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            e = e + edge_mlp(torch.cat([e, h[:, self.src], h[:, self.dst]], -1))
            agg = torch.zeros_like(h).index_add_(1, self.dst, e)
            h = h + node_mlp(torch.cat([h, agg], -1))
        y = self.dec(h[:, :N]).squeeze(-1)
        return (self.q_mid + self.q_half * torch.tanh(y)).reshape(*shape, N)

    def forward(self, q0, goal, s, T):
        """Joint angles q(s) [..., N] at moments s [...] of the move."""
        b = blend_b(s)
        return q0 + b[..., None] * (self.end_angles(q0, goal, b, T) - q0)

    def end_pose(self, q0, goal, T):
        one = torch.ones(q0.shape[:-1], device=q0.device, dtype=q0.dtype)
        return self.end_angles(q0, goal, one, T)

    def motion(self, q0, goal, s, T):
        """q, qd, qdd [..., k, N] in physical time at moments s [..., k] (kept within [FD, 1 - FD]); q0, goal, T per move."""
        s = s.clamp(FD, 1 - FD)
        k = s.shape[-1]
        ss = torch.stack([s - FD, s, s + FD], -1)                          # [..., k, 3]
        ex = lambda x: x[..., None, None, :].expand(*s.shape, 3, x.shape[-1])
        q = self.forward(ex(q0), ex(goal), ss, T[..., None, None].expand(*s.shape, 3))
        qm, qc, qp = q.unbind(-2)
        Tt = T[..., None, None]
        return qc, (qp - qm) / (2 * FD) / Tt, (qp - 2 * qc + qm) / FD ** 2 / Tt ** 2
