"""Train the 5-joint arm's motion PINN on physics alone: no example trajectories.

Each step draws moves between collision-free least-torque poses from dataset.pt (data.py):
loaded moves (0.5-12 kg, both ends clear of the carried part) and empty ones, lasting
0.5-3 times the joint blend's time. The network's motion goes through the equations of
motion at random moments, and the loss is what physics says about it:

    effort    mean over the move of sum_i (tau_i / tau_max_i)^2
    peak      (soft maximum of |tau_i / tau_max_i| over the move and the joints)^2
    joints    joints within 1 deg of a limit                           (x 100,000)
    clear     within 30 mm of the floor, the bin or the part, or 10 mm of the arm itself:
              mean over the move plus its worst moment                 (x 1,000)

There is no motor-limit term: the motors are sized from the torques these motions need
(size_motors.py), not the motions squeezed under chosen motors. tau_max only scales each
joint's torque in the effort and peak terms.

    python train_pinn.py --device cuda
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from kin import JOINT_MARGIN, LIM, Q_HALF, TAU_MAX, collision_penalty, inverse_dynamics, min_clearance
from motion import T_MIN, MotionPINN5, blend, move_time

HERE = Path(__file__).parent
W_PEAK, W_JOINT, W_CLEAR = 1.0, 1e5, 1e3
BETA = 20.0          # sharpness of the soft maximum


def sample_moves(data, n, device, generator=None):
    r = lambda *s: torch.rand(*s, device=device, generator=generator)
    loaded = r(n) < 0.6
    pick = lambda pool: pool[torch.randint(len(pool), (n,), device=device, generator=generator)]
    i0 = torch.where(loaded, pick(data["loaded"]), pick(data["empty"]))
    i1 = torch.where(loaded, pick(data["loaded"]), pick(data["empty"]))
    q0, q1 = data["q"][i0], data["q"][i1]
    payload = torch.where(loaded, 0.5 + r(n) * 11.5, torch.zeros(n, device=device))
    T = (move_time(q0, q1) * torch.exp(math.log(0.5) + r(n) * math.log(6.0))).clamp(min=T_MIN)
    return q0, q1, T, payload


def physics(q, qd, qdd, payload):
    """Per-moment torque ratios and penalties, [..., K] each (r: [..., K, N])."""
    dev = q.device
    r = inverse_dynamics(q, qd, qdd, payload) / TAU_MAX.to(dev)
    lim, half = LIM.to(dev), Q_HALF.to(dev)
    joints = (torch.relu((q - (lim[:, 1] - JOINT_MARGIN)) / half) ** 2 + torch.relu(((lim[:, 0] + JOINT_MARGIN) - q) / half) ** 2).sum(-1)
    return {"r": r, "effort": (r ** 2).sum(-1), "torque": (torch.relu(r.abs() - 1) ** 2).sum(-1), "joints": joints,
            "clear": collision_penalty(q, payload), "min_clear": min_clearance(q, payload)}


def soft_peak(r):
    """Soft maximum of |r| over moments and joints of each move ([..., K, N] -> [...])."""
    a = r.abs().flatten(-2)
    return torch.logsumexp(BETA * a, -1) / BETA


def loss_of(t):
    peak = soft_peak(t["r"])
    per_move = (t["effort"].mean(-1) + W_PEAK * peak ** 2
                + W_JOINT * t["joints"].mean(-1) + W_CLEAR * (t["clear"].mean(-1) + t["clear"].max(-1).values))
    # the worst sampled moment counts too: a brief brush of the arm against itself would vanish in the mean
    return per_move.mean()


def summarise(t):
    """Per move: effort, true peak over the sampled moments, over limit, joint and clearance checks."""
    return {"effort": t["effort"].mean(-1), "peak": t["r"].abs().flatten(-2).max(-1).values,
            "over": (t["r"].abs() > 1).flatten(-2).any(-1), "joints": t["joints"].max(-1).values > 0,
            "collides": t["min_clear"].min(-1).values < 0, "min_clear": t["min_clear"].min(-1).values}


def evaluate(model, moves, k=64, chunk=100):   # chunks of 100 moves keep the derivative graphs within GPU memory
    out = []
    for i in range(0, len(moves[0]), chunk):
        q0, q1, T, p = (x[i:i + chunk] for x in moves)
        s = torch.linspace(0, 1, k, device=q0.device).expand(len(q0), k)
        ex = lambda x: x[:, None].expand(-1, k, *x.shape[1:])
        q, qd, qdd = model.motion(ex(q0), ex(q1), s, ex(T), ex(p))
        out.append(summarise(physics(q.detach(), qd.detach(), qdd.detach(), ex(p))))
    return {key: torch.cat([o[key] for o in out]) for key in out[0]}


def evaluate_blend(moves, k=64):
    q0, q1, T, p = moves
    override = (move_time(q0, q1) / T).clamp(min=0.05)
    t = torch.linspace(0, 1, k, device=q0.device) * T[:, None]
    q, qd, qdd = blend(q0, q1, t, override)
    return summarise(physics(q, qd, qdd, p[:, None].expand(-1, k)))


def report(s):
    return {"effort": s["effort"].mean().item(), "peak_pct": s["peak"].mean().item() * 100,
            "over_limit_pct": s["over"].float().mean().item() * 100, "collides_pct": s["collides"].float().mean().item() * 100,
            "past_joint_margin_pct": s["joints"].float().mean().item() * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--times", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init", default="", help="checkpoint to fine-tune from")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)
    raw = torch.load(HERE / "dataset.pt")
    data = {"q": raw["q"].to(dev)}
    pl = raw["payload"].to(dev)
    data["loaded"], data["empty"] = (pl > 0).nonzero()[:, 0], (pl == 0).nonzero()[:, 0]
    model = MotionPINN5().to(dev)
    if args.init:
        model.load_state_dict(torch.load(HERE / args.init, map_location=dev)["state"])
        print(f"fine-tuning from {args.init}")
    print(f"device: {dev}  parameters: {sum(p.numel() for p in model.parameters()):,}  "
          f"endpoint poses: {len(data['loaded']):,} loaded, {len(data['empty']):,} empty")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    val = sample_moves(data, 2000, dev, torch.Generator(device=dev).manual_seed(1234))
    ref = report(evaluate_blend(val))
    print("joint blend on the validation moves:", {k: round(v, 3) for k, v in ref.items()})
    best, history, t0 = float("inf"), [], time.time()
    for step in range(1, args.steps + 1):
        model.train()
        q0, q1, T, p = sample_moves(data, args.batch, dev)
        k = args.times
        s = torch.rand(args.batch, k, device=dev)
        ex = lambda x: x[:, None].expand(-1, k, *x.shape[1:])
        q, qd, qdd = model.motion(ex(q0), ex(q1), s, ex(T), ex(p))
        loss = loss_of(physics(q, qd, qdd, ex(p)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == args.steps:
            model.eval()
            v = evaluate(model, val)
            rv = report(v)
            score = rv["effort"] + W_PEAK * (v["peak"] ** 2).mean().item() + 10 * rv["collides_pct"] / 100
            history.append({"step": step, "loss": loss.item(), "score": score, **rv})
            print(f"step {step:6d}  loss {loss.item():.4f}  effort {rv['effort']:.3f} (blend {ref['effort']:.3f})  "
                  f"peak {rv['peak_pct']:.0f} % (blend {ref['peak_pct']:.0f} %)  over limit {rv['over_limit_pct']:.1f} %  "
                  f"collides {rv['collides_pct']:.1f} % (blend {ref['collides_pct']:.1f} %)  [{time.time() - t0:4.0f}s]", flush=True)
            if score < best:
                best = score
                torch.save({"state": model.state_dict()}, HERE / "arm5_pinn.pt")
    (HERE / "pinn_history.json").write_text(json.dumps(history))
    print(f"saved best checkpoint to {HERE / 'arm5_pinn.pt'}")


if __name__ == "__main__":
    main()
