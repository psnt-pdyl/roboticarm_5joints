"""Graph network for the N-joint arm's inverse kinematics.

A graph network for any chain length: nodes are the N joints, the tool
point and a goal node. Chain edges carry each link's length; the goal links to every
chain node. Each joint node decodes its own angle, kept inside its limits by tanh.
The goal also carries the payload, because the pose chosen depends on it.
"""
import torch
from torch import nn

from kin import LEN_MM, N, Q_HALF, Q_MID, REACH

N_NODES = N + 2
GOAL = N + 1
_EDGES = []
for k in range(N):
    _EDGES += [(k, k + 1, 0, float(LEN_MM[k]) / REACH), (k + 1, k, 1, float(LEN_MM[k]) / REACH)]
for i in range(N + 1):
    _EDGES += [(GOAL, i, 2, 0.0), (i, GOAL, 3, 0.0)]
SRC, DST = [e[0] for e in _EDGES], [e[1] for e in _EDGES]
GOAL_IN = 5                       # x, z, sin, cos, payload
NODE_IN = N_NODES + GOAL_IN
EDGE_IN = 4 + 1


def static_features():
    node = torch.zeros(N_NODES, NODE_IN)
    node[torch.arange(N_NODES), torch.arange(N_NODES)] = 1.0
    edge = torch.zeros(len(_EDGES), EDGE_IN)
    for k, (_, _, t, length) in enumerate(_EDGES):
        edge[k, t], edge[k, 4] = 1.0, length
    return node, edge


class MLP(nn.Module):
    def __init__(self, i, h, o, norm=True):
        super().__init__()
        self.l1, self.l2 = nn.Linear(i, h), nn.Linear(h, o)
        self.norm = nn.LayerNorm(o) if norm else None

    def forward(self, x):
        x = self.l2(nn.functional.silu(self.l1(x)))
        return self.norm(x) if self.norm is not None else x


class ChainGNN(nn.Module):
    def __init__(self, hidden=128, steps=6):
        super().__init__()
        self.hidden, self.steps = hidden, steps
        node, edge = static_features()
        self.register_buffer("node0", node)
        self.register_buffer("edge0", edge)
        self.register_buffer("src", torch.tensor(SRC))
        self.register_buffer("dst", torch.tensor(DST))
        self.register_buffer("q_mid", Q_MID.clone())
        self.register_buffer("q_half", Q_HALF.clone())
        self.enc_node = MLP(NODE_IN, hidden, hidden)
        self.enc_edge = MLP(EDGE_IN, hidden, hidden)
        self.edge_mlps = nn.ModuleList(MLP(3 * hidden, hidden, hidden) for _ in range(steps))
        self.node_mlps = nn.ModuleList(MLP(2 * hidden, hidden, hidden) for _ in range(steps))
        self.dec = MLP(hidden, hidden, 1, norm=False)

    def forward(self, goal):
        """goal: [B, 5] goal features -> joint angles [B, N] in radians."""
        B = goal.shape[0]
        x = self.node0.expand(B, -1, -1).clone()
        x[:, GOAL, N_NODES:] = goal
        h = self.enc_node(x)
        e = self.enc_edge(self.edge0).expand(B, -1, -1)
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            e = e + edge_mlp(torch.cat([e, h[:, self.src], h[:, self.dst]], -1))
            agg = torch.zeros_like(h).index_add_(1, self.dst, e)
            h = h + node_mlp(torch.cat([h, agg], -1))
        y = self.dec(h[:, :N]).squeeze(-1)
        return self.q_mid + self.q_half * torch.tanh(y)
