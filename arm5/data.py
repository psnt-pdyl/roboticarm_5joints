"""Search for the least-torque pose of each target, as training labels for the GNN.

For every target (tool point, tool angle, payload), run projected damped least squares
from K random starting poses. Keep the attempts that reach the target (within 0.5 mm
and 0.1 deg) without collision (every link 15 mm clear of the floor and the bin, and of
the carried part when there is a payload, and every part of the arm 5 mm clear of every
other part and of the pedestal), and label the target with the one that needs
the least torque to hold its payload still. Targets no attempt reaches are dropped.

    python data.py --targets 1000000 --starts 16       # writes dataset.pt
"""
import argparse
import time
from pathlib import Path

import torch

from kin import D, FLOOR_CLEAR, LIM, N, Q_MID, SELF_MARGIN, clearance, dls_fast, fk, holding, points, sample_targets, self_clearance

HERE = Path(__file__).parent


def search(tcp, phi, payload, starts, steps, generator=None):
    """Best reachable least-torque pose per target [B, N] and a mask of targets that have one."""
    B, dev = len(tcp), tcp.device
    lim = LIM.to(dev)
    q0 = lim[:, 0] + torch.rand(B, starts, N, device=dev, generator=generator) * (lim[:, 1] - lim[:, 0])
    q0[:, 0] = Q_MID.to(dev)                                           # one attempt from the rest pose
    ex = lambda x: x[:, None].expand(-1, starts, *x.shape[1:])
    q = dls_fast(q0, ex(tcp), ex(phi), steps)
    p, a = fk(q)
    ok = ((p - ex(tcp)).norm(dim=-1) < 0.5) & ((torch.atan2((a - ex(phi)).sin(), (a - ex(phi)).cos()) / D).abs() < 0.1)
    ok &= (points(q)[..., 1:, 1] >= FLOOR_CLEAR).all(-1)
    ok &= clearance(q, ex(payload)).min(-1).values >= 15.0     # clear of floor, bin and the carried part
    ok &= self_clearance(q).min(-1).values >= SELF_MARGIN / 2   # and of the arm's own links, housings, gripper and pedestal
    effort = (holding(q, ex(payload)) ** 2).sum(-1).masked_fill(~ok, float("inf"))
    best = effort.argmin(-1)
    return q[torch.arange(B, device=dev), best], ok.any(-1), effort.min(-1).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, default=1_000_000)
    ap.add_argument("--starts", type=int, default=16)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    g = torch.Generator(device=args.device).manual_seed(7)
    keep = {"tcp": [], "phi": [], "payload": [], "q": [], "effort": []}
    found, t0 = 0, time.time()
    for i in range(0, args.targets, args.chunk):
        tcp, phi, payload, q_gen = sample_targets(args.chunk, args.device, g)
        q, ok, eff = search(tcp, phi, payload, args.starts, args.steps, g)
        for k, v in (("tcp", tcp), ("phi", phi), ("payload", payload), ("q", q), ("effort", eff)):
            keep[k].append(v[ok].cpu())
        found += int(ok.sum())
        gen = (holding(q_gen, payload) ** 2).sum(-1)
        print(f"{i + args.chunk:8d} targets  labelled {found:8d}  search effort {eff[ok].mean():.3f} "
              f"(generating pose {gen[ok].mean():.3f})  [{time.time() - t0:4.0f}s]", flush=True)
    data = {k: torch.cat(v) for k, v in keep.items()}
    torch.save(data, HERE / "dataset.pt")
    print(f"wrote {HERE / 'dataset.pt'}: {len(data['q']):,} labelled targets")


if __name__ == "__main__":
    main()
