"""Train the motion GNN (motion_gnn.py): one graph network for the whole move, on the pose GNN's and the PINN's losses.

Each step: targets with their searched least-torque labels, and start poses from other targets of the same kind
(loaded or empty), moves lasting 0.5-3 times the joint blend's time. The loss is

    L = L_pose(q(1))  +  w_motion * L_motion(q(s) at random moments)      (w_motion 5, label weight 2 by default)

L_pose is the pose GNN's loss on where the move ends (reach, label, holding torque, joint limits, floor, collisions);
L_motion is the motion PINN's (effort, peak torque, joint limits, collisions: mean and worst moment). Starts from the
pose GNN, i.e. from a plain smooth move to its pose. Validates on the 20,000 held-out targets.

    python train_mgnn.py --steps 10000
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from kin import goal_features
from motion import T_MIN, move_time
from motion_gnn import MotionGNN
from train import loss_of as pose_loss, pose_terms, weights as pose_weights
from train_pinn import W_JOINT, physics, soft_peak

MOTION_W = {"effort": 1.0, "peak": 1.0, "clear": 1e3}       # the PINN's weights; --w-effort etc. change them


def motion_loss(t, w=MOTION_W):
    """The motion PINN's loss (train_pinn.loss_of) with adjustable weights on effort, peak torque and clearance."""
    peak = soft_peak(t["r"])
    per_move = (w["effort"] * t["effort"].mean(-1) + w["peak"] * peak ** 2 + W_JOINT * t["joints"].mean(-1)
                + w["clear"] * (t["clear"].mean(-1) + t["clear"].max(-1).values))
    return per_move.mean()

HERE = Path(__file__).parent
N_VAL = 20000


def sample(data, n, device, lo, hi, generator=None):
    """Targets from rows [lo, hi) with labels, start poses of the same kind (loaded or empty), durations."""
    r = lambda *s: torch.rand(*s, device=device, generator=generator)
    i = torch.randint(lo, hi, (n,), device=device, generator=generator)
    loaded = data["payload"][i] > 0
    pick = lambda pool: pool[torch.randint(len(pool), (n,), device=device, generator=generator)]
    j = torch.where(loaded, pick(data["loaded"]), pick(data["empty"]))
    q0, label = data["q"][j], data["q"][i]
    T = (move_time(q0, label) * torch.exp(math.log(0.5) + r(n) * math.log(6.0))).clamp(min=T_MIN)
    return data["tcp"][i], data["phi"][i], data["payload"][i], label, q0, T


def losses(model, batch, k, pose_w, generator=None, motion_w=MOTION_W):
    tcp, phi, payload, label, q0, T = batch
    goal = goal_features(tcp, phi, payload)
    t = pose_terms(model.end_pose(q0, goal, T), tcp, phi, payload)
    lp = pose_loss(t, pose_w, label)
    s = torch.rand(len(q0), k, device=q0.device, generator=generator)
    q, qd, qdd = model.motion(q0, goal, s, T)
    ex = lambda x: x[:, None].expand(-1, k, *x.shape[1:])
    ph = physics(q, qd, qdd, ex(payload))
    return lp, motion_loss(ph, motion_w), t, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--times", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default="arm5_gnn.pt", help="pose GNN to start from")
    ap.add_argument("--out", default="arm5_mgnn.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--w-motion", type=float, default=5.0, help="weight of the motion loss")
    ap.add_argument("--label-weight", type=float, default=2.0, help="weight of the least-torque label in the pose loss")
    ap.add_argument("--time-goal-only", action="store_true", help="first version: moment and duration on the goal node only")
    ap.add_argument("--resume", default="", help="motion GNN checkpoint to continue from (instead of the pose GNN)")
    ap.add_argument("--w-effort", type=float, default=1.0)
    ap.add_argument("--w-peak", type=float, default=1.0)
    ap.add_argument("--w-clear", type=float, default=1e3, help="clearance-margin weight")
    args = ap.parse_args()
    W_MOTION = args.w_motion
    pose_w = {**pose_weights(1.0, True), "label": args.label_weight}
    dev = args.device
    torch.manual_seed(0)
    raw = torch.load(HERE / "dataset.pt")
    data = {k: v.to(dev) for k, v in raw.items()}
    train_rows = torch.arange(N_VAL, len(data["q"]), device=dev)
    data["loaded"] = train_rows[data["payload"][N_VAL:] > 0]
    data["empty"] = train_rows[data["payload"][N_VAL:] == 0]
    ck = torch.load(HERE / args.init, map_location=dev)
    if args.resume:
        cr = torch.load(HERE / args.resume, map_location=dev)
        model = MotionGNN(cr["hidden"], cr["steps"], cr.get("time_all", False)).to(dev)
        model.load_state_dict(cr["state"])
    else:
        model = MotionGNN.from_pose_gnn(ck["state"], ck["hidden"], ck["steps"], not args.time_goal_only).to(dev)
    motion_w = {"effort": args.w_effort, "peak": args.w_peak, "clear": args.w_clear}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    g = torch.Generator(device=dev).manual_seed(4321)
    val = sample(data, 1000, dev, 0, N_VAL, g)                       # held-out targets
    val_s = torch.Generator(device=dev).manual_seed(99)
    print(f"motion GNN from {args.resume or args.init} (motion weights {motion_w}): {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{len(train_rows):,} training targets", flush=True)

    def validate():
        model.eval()
        lp, lm, reach, coll_end, peak, effort, coll_path = [], [], [], [], [], [], []
        val_s.manual_seed(99)
        with torch.no_grad():
            for c in range(0, 1000, 250):
                b = tuple(x[c:c + 250] for x in val)
                a, m, t, ph = losses(model, b, 32, pose_w, val_s, motion_w)
                lp.append(a.item()); lm.append(m.item())
                reach.append(t["pos"]); coll_end.append(t["collides"].float())
                peak.append(ph["r"].abs().flatten(-2).max(-1).values); effort.append(ph["effort"].mean(-1))
                coll_path.append((ph["min_clear"].min(-1).values < 0).float())
        cat = lambda xs: torch.cat(xs)
        return {"pose": sum(lp) / len(lp), "motion": sum(lm) / len(lm), "reach_mm": cat(reach).mean().item(),
                "end_collides_pct": cat(coll_end).mean().item() * 100, "effort": cat(effort).mean().item(),
                "peak_pct": cat(peak).mean().item() * 100, "path_collides_pct": cat(coll_path).mean().item() * 100}

    v = validate()
    best, history, t0 = v["pose"] + W_MOTION * v["motion"], [{"step": 0, **v}], time.time()
    print(f"start: {json.dumps({k: round(x, 3) for k, x in v.items()})}", flush=True)
    torch.save({"state": model.state_dict(), "hidden": model.hidden, "steps": model.steps, "time_all": model.time_all}, HERE / args.out)
    for step in range(1, args.steps + 1):
        model.train()
        lp, lm, _, _ = losses(model, sample(data, args.batch, dev, N_VAL, len(data["q"])), args.times, pose_w, motion_w=motion_w)
        loss = lp + W_MOTION * lm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            v = validate()
            history.append({"step": step, **v})
            total = v["pose"] + W_MOTION * v["motion"]
            print(f"step {step:6d}  pose {v['pose']:.3f}  motion {v['motion']:.3f}  total {total:.3f}  reach {v['reach_mm']:.2f} mm  "
                  f"effort {v['effort']:.3f}  peak {v['peak_pct']:.0f} %  collides: end {v['end_collides_pct']:.1f} %, "
                  f"path {v['path_collides_pct']:.1f} %  [{time.time() - t0:4.0f}s]", flush=True)
            if total < best:
                best = total
                torch.save({"state": model.state_dict(), "hidden": model.hidden, "steps": model.steps, "time_all": model.time_all}, HERE / args.out)
    (HERE / "mgnn_history.json").write_text(json.dumps(history))
    print(f"saved best to {args.out} (total {best:.3f})")


if __name__ == "__main__":
    main()
