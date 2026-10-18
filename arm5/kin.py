"""The 5-joint planar arm: geometry, limits, masses, kinematics and statics.

780 mm reach split over five links, with motors that shrink
toward the tool; their ratings (and so the links' masses) come from size_motors.py. Units: mm and radians for kinematics, SI for
torques. A target (x, z, tool angle) leaves two joints free, so there are infinitely
many poses for it; train.py teaches the GNN to pick the one needing the least torque.
"""
import math

import torch

D = math.pi / 180
N = 5
LEN_MM = torch.tensor([220.0, 180.0, 150.0, 110.0, 120.0])   # last link: wrist to tool point
REACH = float(LEN_MM.sum())                                  # 780 mm
SH = (0.0, 160.0)                                            # shoulder position, mm
LIM = torch.tensor([[-20.0, 200.0]] + [[-150.0, 150.0]] * 4) * D
Q_MID = LIM.mean(1)
Q_HALF = (LIM[:, 1] - LIM[:, 0]) / 2
# Masses. Each link was 4, 3, 2, 1.5, 1 kg with the first motors (80, 45, 25, 12, 6 N m), 40 % of it the motor.
# Now a link is its structure (the other 60 %, centred mid-link) plus the motor of the next joint, which sits at
# the link's far end; the A1 motor sits in the pedestal and loads no joint. A motor's mass grows with its rated
# torque (kg per N m as the first motors), so stronger motors make the arm heavier: arm_masses(tau).
TAU_MAX_OLD = torch.tensor([80.0, 45.0, 25.0, 12.0, 6.0])
STRUCT_MASS = torch.tensor([4.0, 3.0, 2.0, 1.5, 1.0]) * 0.6
MOTOR_KG_PER_NM = torch.tensor([4.0, 3.0, 2.0, 1.5, 1.0]) * 0.4 / TAU_MAX_OLD


def arm_masses(tau):
    """Per link: mass (kg), centre of mass from its joint (m), inertia about that centre (kg m^2), for motors rated tau."""
    L = LEN_MM / 1000
    tip = torch.cat([MOTOR_KG_PER_NM[1:] * tau[1:], torch.zeros(1)])   # motor k+1 at the end of link k
    m = STRUCT_MASS + tip
    c = (STRUCT_MASS * L / 2 + tip * L) / m
    inertia = STRUCT_MASS * L ** 2 / 12 + STRUCT_MASS * (L / 2 - c) ** 2 + tip * (L - c) ** 2
    return m, c, inertia


TAU_MAX = torch.tensor([297.0, 154.0, 111.0, 68.0, 32.0])    # N m: size_motors.py --factor 3, every move doable at 3x the blend time
MASS, LINK_COM, LINK_I = arm_masses(TAU_MAX)
PAYLOAD_MAX = 12.0                                           # kg, training range
PART_MASS = 2.0                                              # kg, the page's default part
G = 9.81
FLOOR_CLEAR = 20.0                                           # mm
JOINT_MARGIN = 1.0 * D


def points(q):
    """Joint positions [..., N + 1, 2] in mm (shoulder first, tool point last)."""
    a = q.cumsum(-1)
    xy = (LEN_MM.to(q.device)[:, None] * torch.stack([a.cos(), a.sin()], -1)).cumsum(-2)
    sh = torch.tensor(SH, device=q.device).expand(*q.shape[:-1], 1, 2)
    return torch.cat([sh, sh + xy], -2)


def fk(q):
    """Tool point [..., 2] (mm) and tool angle [...]."""
    return points(q)[..., -1, :], q.sum(-1)


def inverse_dynamics(q, qd, qdd, payload):
    """Motor torques [..., N] in N m. Planar recursive Newton-Euler over N links, payload at the tool point."""
    dev = q.device
    L, m = LEN_MM.to(dev) / 1000, MASS.to(dev)
    c, I = LINK_COM.to(dev), LINK_I.to(dev)
    a, ad, add = q.cumsum(-1), qd.cumsum(-1), qdd.cumsum(-1)
    u = torch.stack([a.cos(), a.sin()], -1)
    n = torch.stack([-a.sin(), a.cos()], -1)
    rel = add[..., None] * n - (ad ** 2)[..., None] * u
    g = torch.tensor([0.0, -G], device=dev)
    cross = lambda r, f: r[..., 0] * f[..., 1] - r[..., 1] * f[..., 0]
    p_acc = [torch.zeros_like(u[..., 0, :])]
    for k in range(N):
        p_acc.append(p_acc[-1] + L[k] * rel[..., k, :])
    F = payload[..., None] * (p_acc[N] - g)
    tau_next = torch.zeros_like(q[..., 0])
    taus = [None] * N
    for k in reversed(range(N)):
        F_k = m[k] * (p_acc[k] + c[k] * rel[..., k, :] - g) + F
        taus[k] = I[k] * add[..., k] + tau_next + c[k] * cross(u[..., k, :], F_k) + (L[k] - c[k]) * cross(u[..., k, :], F)
        F, tau_next = F_k, taus[k]
    return torch.stack(taus, -1)


LINK_R = torch.tensor([29.0, 24.0, 20.0, 16.0, 14.0])        # mm, link thickness as drawn
PART_R = 60.0 / math.sqrt(2)                                 # mm, the 60 mm part as a circle around its corners
BIN = (-185.0, 120.0, 130.0)                                 # bin centre x, width, height (mm), fixed in the cell
COLL_MARGIN = 30.0


def clearance(q, payload):
    """Signed clearances [..., K] in mm (negative = collision) of circles along the links from the floor,
    the bin, and - when carrying (payload > 0) - the part held at the tool point (the gripper's own link
    excluded). Same rules as penalty() in arm5.html."""
    P = points(q)
    t = torch.tensor([0.25, 0.5, 0.75, 1.0], device=q.device)
    a, b = P[..., :-1, None, :], P[..., 1:, None, :]
    c = a + (b - a) * t[:, None]                                         # [..., N, 4, 2]
    r = LINK_R.to(q.device)[:, None]
    floor = c[..., 1] - r
    bx, bw, bh = BIN
    dx = c[..., 0] - c[..., 0].clamp(bx - bw / 2, bx + bw / 2)
    dy = c[..., 1] - c[..., 1].clamp(0.0, bh)
    binc = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6) - r
    tool = P[..., -1, None, None, :]
    part = (c - tool).norm(dim=-1) - r - PART_R
    part = torch.where((payload > 0)[..., None, None], part, torch.full_like(part, 1e3))[..., :-1, :]
    return torch.cat([floor.flatten(-2), binc.flatten(-2), part.flatten(-2)], -1)


JOINT_R = torch.tensor([44.0, 36.0, 30.0, 25.0, 21.0])       # mm, joint housings as drawn
SELF_MARGIN = 10.0                                           # mm, how close two parts of the arm may come
BASE = ((-80.0, 0.0), (80.0, 0.0), (52.0, SH[1] - 30), (-52.0, SH[1] - 30))   # the pedestal, a trapezoid


def _seg_dist(p, a, b):
    """Distance [...] from points p to segments a-b."""
    d = b - a
    t = (((p - a) * d).sum(-1) / (d * d).sum(-1).clamp(min=1e-9)).clamp(0, 1)
    return (p - a - t[..., None] * d).norm(dim=-1)


def _crosses(a, b, c, d):
    cr = lambda o, u, v: (u[..., 0] - o[..., 0]) * (v[..., 1] - o[..., 1]) - (u[..., 1] - o[..., 1]) * (v[..., 0] - o[..., 0])
    return (cr(a, b, c) * cr(a, b, d) < 0) & (cr(c, d, a) * cr(c, d, b) < 0)


def seg_seg(a, b, c, d):
    """Signed distance [...] between segments a-b and c-d: their gap, or minus the smaller endpoint depth when they cross."""
    m = torch.stack([_seg_dist(a, c, d), _seg_dist(b, c, d), _seg_dist(c, a, b), _seg_dist(d, a, b)], -1).min(-1).values
    return torch.where(_crosses(a, b, c, d), -m, m)


def bodies(q):
    """The arm's solid parts as capsules: (start [..., K, 2], end [..., K, 2], radius [K], link index [K]).
    Links A1-A4 and the joint housings as drawn in arm5.html; the gripper as its crossbar and two fingers."""
    P = points(q)
    dev = q.device
    w, tool = P[..., N - 1, :], P[..., N, :]
    u = (tool - w) / LEN_MM[N - 1]
    n = torch.stack([-u[..., 1], u[..., 0]], -1)
    at = lambda a, b: w + u * a + n * b
    L5 = float(LEN_MM[N - 1])
    starts = [P[..., k, :] for k in range(N - 1)] + [P[..., k, :] for k in range(N)] + [at(L5 - 62, -64), at(L5 - 50, -61), at(L5 - 50, 61)]
    ends = [P[..., k + 1, :] for k in range(N - 1)] + [P[..., k, :] for k in range(N)] + [at(L5 - 62, 64), at(L5 + 12, -61), at(L5 + 12, 61)]
    radius = torch.cat([LINK_R[:N - 1], JOINT_R, torch.tensor([10.0, 6.0, 6.0])]).to(dev)
    link = [0, 1, 2, 3] + [k - 0.5 for k in range(N)] + [4, 4, 4]    # a housing sits between the two links it joins
    kind = ["link"] * 4 + ["joint"] * 5 + ["gripper"] * 3
    return torch.stack(starts, -2), torch.stack(ends, -2), radius, link, kind


def _self_pairs():
    """Pairs of bodies that can collide: parts of links that are not neighbours, plus the gripper against the forearm
    (A4's link). A link and the housings at its two ends always touch, as do a link's two housings, so they are left out."""
    _, _, _, link, kind = bodies(torch.zeros(N))
    pairs = []
    for i in range(len(link)):
        for j in range(i + 1, len(link)):
            gap = abs(link[i] - link[j])
            grip = "gripper" in (kind[i], kind[j]) and min(link[i], link[j]) == 3
            if gap >= 1.5 or grip:
                pairs.append((i, j))
    return torch.tensor(pairs)


SELF_PAIRS = _self_pairs()


def _base_dist(p):
    """Distance [...] from points p to the pedestal (0 inside)."""
    y = p[..., 1].clamp(0.0, BASE[2][1])
    half = BASE[1][0] + (BASE[2][0] - BASE[1][0]) * y / BASE[2][1]
    dx = (p[..., 0].abs() - half).clamp(min=0)
    dy = (p[..., 1] - BASE[2][1]).clamp(min=0) + (-p[..., 1]).clamp(min=0)
    return torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)


def self_clearance(q):
    """Signed gaps [..., K] in mm (negative = collision) between parts of the arm that can meet, and between the
    arm and its pedestal (all but the first link and the shoulder housing, which are mounted on it)."""
    a, b, r, link, _ = bodies(q)
    pr = SELF_PAIRS.to(q.device)
    i, j = pr[:, 0], pr[:, 1]
    gaps = seg_seg(a[..., i, :], b[..., i, :], a[..., j, :], b[..., j, :]) - r[i] - r[j]
    keep = torch.tensor([k for k, l in enumerate(link) if l >= 1], device=q.device)
    s = torch.linspace(0, 1, 5, device=q.device)
    pts = a[..., keep, None, :] + (b - a)[..., keep, None, :] * s[:, None]
    base = (_base_dist(pts) - r[keep][:, None]).flatten(-2)
    return torch.cat([gaps, base], -1)


def min_clearance(q, payload):
    """Closest any part of the arm comes to the cell (floor, bin, carried part) or to another part of itself, in mm."""
    return torch.minimum(clearance(q, payload).min(-1).values, self_clearance(q).min(-1).values)


def collision_penalty(q, payload, margin=COLL_MARGIN):
    """Sum of squared intrusions into the margins (the cell's, and SELF_MARGIN between the arm's own parts), in units of 100 mm."""
    return ((torch.relu(margin - clearance(q, payload)) / 100).pow(2).sum(-1)
            + (torch.relu(SELF_MARGIN - self_clearance(q)) / 100).pow(2).sum(-1))


def holding(q, payload):
    """Torque [..., N] needed to hold pose q still with the payload, as a fraction of each motor's limit."""
    z = torch.zeros_like(q)
    return inverse_dynamics(q, z, z, payload) / TAU_MAX.to(q.device)


def sample_targets(n, device="cpu", generator=None):
    """Reachable targets: tool point (x, z) mm and tool angle, from random poses inside the limits
    that keep every joint 20 mm above the floor. Payload uniform 0..PAYLOAD_MAX (a quarter of them 0)."""
    lim = LIM.to(device)
    q = torch.empty(0, N, device=device)
    while len(q) < n:
        c = lim[:, 0] + torch.rand(3 * n, N, device=device, generator=generator) * (lim[:, 1] - lim[:, 0])
        q = torch.cat([q, c[(points(c)[..., 1:, 1] >= FLOOR_CLEAR).all(-1)]])
    q = q[:n]
    tcp, phi = fk(q)
    r = torch.rand(n, device=device, generator=generator)
    payload = torch.where(r < 0.25, torch.zeros_like(r), torch.rand(n, device=device, generator=generator) * PAYLOAD_MAX)
    return tcp, torch.atan2(phi.sin(), phi.cos()), payload, q


def goal_features(tcp, phi, payload):
    return torch.stack([(tcp[..., 0] - SH[0]) / REACH, (tcp[..., 1] - SH[1]) / REACH,
                        phi.sin(), phi.cos(), payload / PAYLOAD_MAX], -1)


def dls_fast(q, tcp, phi, steps=30, lam=5.0):
    """The same projected damped least squares as dls(), with the planar Jacobian written out
    (the tool point moves perpendicular to the line from each joint) instead of autograd,
    so millions of poses can be solved at once."""
    lo, hi = LIM[:, 0].to(q.device), LIM[:, 1].to(q.device)
    eye = lam ** 2 * torch.eye(3, device=q.device)
    for _ in range(steps):
        P = points(q)
        t, a = P[..., -1, :], q.sum(-1)
        e = torch.cat([tcp - t, torch.atan2((phi - a).sin(), (phi - a).cos())[..., None] * 100], -1)
        d = t[..., None, :] - P[..., :-1, :]
        J = torch.stack([-d[..., 1], d[..., 0], torch.full_like(d[..., 0], 100.0)], -2)
        y = torch.linalg.solve(J @ J.transpose(-1, -2) + eye, e[..., None])
        q = torch.max(torch.min(q + (J.transpose(-1, -2) @ y)[..., 0], hi), lo)
    return q


def dls(q, tcp, phi, steps=2, lam=5.0):
    """Damped-least-squares refinement of q toward (tcp, phi): the smallest joint change per step,
    clamped back inside the joint limits after each step."""
    lo, hi = LIM[:, 0].to(q.device), LIM[:, 1].to(q.device)
    for _ in range(steps):
        q = q.detach().requires_grad_(True)
        p, a = fk(q)
        e = torch.cat([tcp - p, torch.atan2((phi - a).sin(), (phi - a).cos())[..., None] * 100], -1)   # 1 rad ~ 100 mm
        J = torch.stack([torch.autograd.grad(torch.cat([p, a[..., None] * 100], -1)[..., i].sum(), q, retain_graph=True)[0]
                         for i in range(3)], -2)                                                            # [..., 3, N]
        JJt = J @ J.transpose(-1, -2) + lam ** 2 * torch.eye(3, device=q.device)
        q = (q + (J.transpose(-1, -2) @ torch.linalg.solve(JJt, e[..., None]))[..., 0]).detach()
        q = torch.max(torch.min(q, hi), lo)
    return q
