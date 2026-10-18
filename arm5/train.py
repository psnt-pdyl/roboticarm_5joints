"""Train the 5-joint arm's IK graph network with no answer labels.

The arm has two more joints than a target pins down, so there is no single right
answer to copy. The loss says what a good pose is instead:

    reach    (|tool point - target| / 2 mm)^2 + (tool-angle error / 0.5 deg)^2
    gravity  sum_i (holding torque_i / torque limit_i)^2, with the payload     (ramps in)
    limits   joints within 1 deg of a limit, elbow/wrist/tool within 20 mm of the floor

Reach comes first: gravity ramps in over the first 30 % of steps, once the network
can put the tool on target, so it then picks the least-torque pose among those.

    python train.py --device cuda
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from kin import (D, FLOOR_CLEAR, JOINT_MARGIN, LIM, Q_HALF, Q_MID, collision_penalty, min_clearance, fk, goal_features,
                 holding, points, sample_targets)
from model import ChainGNN

HERE = Path(__file__).parent
W_LIMIT, W_FLOOR, W_COLL = 1e4, 1e3, 1e3


def terms(model, tcp, phi, payload):
    return pose_terms(model(goal_features(tcp, phi, payload)), tcp, phi, payload)


def pose_terms(q, tcp, phi, payload):
    """The pose loss's terms for joint angles q reaching for the target (tcp, phi) with the payload."""
    p, a = fk(q)
    pos = (p - tcp).norm(dim=-1)
    tool = torch.atan2((a - phi).sin(), (a - phi).cos()) / D
    grav = (holding(q, payload) ** 2).sum(-1)
    lim, half = LIM.to(q.device), Q_HALF.to(q.device)
    joints = (torch.relu((q - (lim[:, 1] - JOINT_MARGIN)) / half) ** 2
              + torch.relu(((lim[:, 0] + JOINT_MARGIN) - q) / half) ** 2).sum(-1)
    floor = (torch.relu((FLOOR_CLEAR - points(q)[..., 1:, 1]) / 100) ** 2).sum(-1)
    coll = collision_penalty(q, payload)
    return {"q": q, "pos": pos, "tool": tool.abs(), "grav": grav, "joints": joints, "floor": floor, "coll": coll,
            "collides": min_clearance(q, payload) < 0}


def huber(x):
    return torch.where(x < 1, 0.5 * x ** 2, x - 0.5)


def weights(frac, labels, finetune=None):
    """Loss weights at this point of training (0..1). Without labels: pull toward the rest pose first,
    then reach, then gravity. With searched least-torque labels: follow them throughout, ramp reach in.
    Fine-tuning a trained model: fixed weights, labels only as a light guide."""
    if finetune:
        return {"label": finetune["label"], "rest": 0.0, "reach": 1.0, "grav": finetune["grav"]}
    if labels:
        return {"label": 10.0, "rest": 0.0, "reach": min(1.0, 0.01 + frac / 0.3),
                "grav": 0.5 * min(1.0, max(0.0, (frac - 0.3) / 0.2))}
    return {"label": 0.0, "rest": max(0.0, 1 - frac / 0.4), "reach": min(1.0, 0.01 + frac / 0.3),
            "grav": min(1.0, max(0.0, (frac - 0.3) / 0.2))}


def loss_of(t, w, q_label=None):
    reach = huber(t["pos"] / 2) + huber(t["tool"] / 0.5)
    half = Q_HALF.to(t["q"].device)
    rest = (((t["q"] - Q_MID.to(t["q"].device)) / half) ** 2).sum(-1)
    label = (((t["q"] - q_label) / half) ** 2).sum(-1) if q_label is not None else 0.0
    return (w["reach"] * reach + w["rest"] * rest + w["label"] * label + w["grav"] * t["grav"]
            + W_LIMIT * t["joints"] + W_FLOOR * t["floor"] + W_COLL * t["coll"]).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--mp-steps", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data", default="dataset.pt", help="searched least-torque labels from data.py ('' for none)")
    ap.add_argument("--init", default="", help="checkpoint to fine-tune from")
    ap.add_argument("--label-weight", type=float, default=1.0, help="fine-tuning: label weight")
    ap.add_argument("--grav-weight", type=float, default=1.0, help="fine-tuning: gravity weight")
    ap.add_argument("--out", default="arm5_gnn.pt")
    args = ap.parse_args()
    ft = {"label": args.label_weight, "grav": args.grav_weight} if args.init else None

    torch.manual_seed(args.seed)
    dev = args.device
    model = ChainGNN(args.hidden, args.mp_steps).to(dev)
    if args.init:
        model.load_state_dict(torch.load(HERE / args.init, map_location=dev)["state"])
        print(f"fine-tuning from {args.init}: label weight {args.label_weight}, gravity weight {args.grav_weight}")
    print(f"device: {dev}  parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    data = torch.load(HERE / args.data) if args.data and (HERE / args.data).exists() else None
    if data is not None:
        data = {k: v.to(dev) for k, v in data.items()}
        n_val = 20000
        val = (data["tcp"][:n_val], data["phi"][:n_val], data["payload"][:n_val])
        val_label, n_all = data["q"][:n_val], len(data["q"])
        print(f"labels: {n_all - n_val:,} searched least-torque poses (mean holding effort {data['effort'][n_val:].mean():.3f})")
    else:
        val, val_label = sample_targets(20000, dev, torch.Generator(device=dev).manual_seed(1234))[:3], None
    best, history, ckpt, t0 = float("inf"), [], HERE / args.out, time.time()
    for step in range(1, args.steps + 1):
        model.train()
        w = weights(step / args.steps, data is not None, ft)
        if data is not None:
            i = torch.randint(20000, n_all, (args.batch,), device=dev)
            loss = loss_of(terms(model, data["tcp"][i], data["phi"][i], data["payload"][i]), w, data["q"][i])
        else:
            loss = loss_of(terms(model, *sample_targets(args.batch, dev)[:3]), w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                t = terms(model, *val)
            vloss = loss_of(t, weights(1.0, data is not None, ft), val_label).item()
            row = {"step": step, "loss": loss.item(), "val_loss": vloss, "pos_mean_mm": t["pos"].mean().item(),
                   "pos_p95_mm": t["pos"].quantile(0.95).item(), "tool_mean_deg": t["tool"].mean().item(),
                   "grav": t["grav"].mean().item(), "past_limit_pct": (t["joints"] > 0).float().mean().item() * 100,
                   "collides_pct": t["collides"].float().mean().item() * 100}
            history.append(row)
            print(f"step {step:6d}  loss {row['loss']:.4f}  val {vloss:.4f}  tool point {row['pos_mean_mm']:6.2f} mm "
                  f"(p95 {row['pos_p95_mm']:6.2f})  tool {row['tool_mean_deg']:5.2f}°  gravity {row['grav']:.4f}  "
                  f"near limit {row['past_limit_pct']:4.1f} %  collides {row['collides_pct']:4.1f} %  [{time.time() - t0:4.0f}s]", flush=True)
            if (ft or step >= 0.5 * args.steps) and vloss < best:   # once the loss has reached its final form
                best = vloss
                torch.save({"state": model.state_dict(), "hidden": args.hidden, "steps": args.mp_steps}, ckpt)
    (HERE / ("train_history_finetune.json" if ft else "train_history.json")).write_text(json.dumps(history))
    print(f"saved best checkpoint to {ckpt} (val loss {best:.4f})")


if __name__ == "__main__":
    main()
