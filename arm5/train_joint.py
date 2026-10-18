"""Train the pose GNN and the motion PINN together, each on both losses.

Each step: a start pose q0 and a target (tool point, tool angle, payload) with its searched
least-torque label. The GNN proposes the end pose q1 = GNN(target); the PINN plans the move
q0 -> q1. One loss covers both:

    L = L_pose(q1) + W_MOTION * L_motion(q0 -> q1)

L_pose is the GNN's loss (reach, label, holding torque, joint limits, floor, collisions);
L_motion is the PINN's (effort, peak torque, torque limits, joint limits, collisions along
the path). Gradients from both reach both networks: through q1 the GNN learns which end
poses are costly or collision-prone to move to, and the PINN trains on the end poses the
GNN actually produces. Starts from the separately trained models.

    python train_joint.py --steps 10000
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from model import ChainGNN
from motion import T_MIN, MotionPINN5, move_time
from train import loss_of as pose_loss, terms as pose_terms, weights as pose_weights
from train_pinn import loss_of as motion_loss, physics

HERE = Path(__file__).parent
W_MOTION = 1.0


def sample(data, n, device, generator=None):
    """Targets with their labels, and start poses of the same kind (loaded or empty) from other targets."""
    r = lambda *s: torch.rand(*s, device=device, generator=generator)
    i = torch.randint(20000, data["n"], (n,), device=device, generator=generator)
    loaded = data["payload"][i] > 0
    pick = lambda pool: pool[torch.randint(len(pool), (n,), device=device, generator=generator)]
    j = torch.where(loaded, pick(data["loaded"]), pick(data["empty"]))
    return data["tcp"][i], data["phi"][i], data["payload"][i], data["q"][i], data["q"][j], r(n)


def step_losses(gnn, pinn, batch, k):
    tcp, phi, payload, q_label, q0, u = batch
    t = pose_terms(gnn, tcp, phi, payload)
    lp = pose_loss(t, pose_weights(1.0, True), q_label)
    q1 = t["q"]                                                   # the GNN's end pose, gradients kept
    T = (move_time(q0, q1.detach()) * torch.exp(math.log(0.5) + u * math.log(6.0))).clamp(min=T_MIN)
    s = torch.rand(len(q0), k, device=q0.device)
    ex = lambda x: x[:, None].expand(-1, k, *x.shape[1:])
    q, qd, qdd = pinn.motion(ex(q0), ex(q1), s, ex(T), ex(payload))
    lm = motion_loss(physics(q, qd, qdd, ex(payload)))
    return lp, lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--times", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)
    raw = torch.load(HERE / "dataset.pt")
    data = {k: v.to(dev) for k, v in raw.items()}
    data["n"] = len(data["q"])
    data["loaded"], data["empty"] = (data["payload"] > 0).nonzero()[:, 0], (data["payload"] == 0).nonzero()[:, 0]

    ck = torch.load(HERE / "arm5_gnn.pt", map_location=dev)
    gnn = ChainGNN(ck["hidden"], ck["steps"]).to(dev)
    gnn.load_state_dict(ck["state"])
    pinn = MotionPINN5().to(dev)
    pinn.load_state_dict(torch.load(HERE / "arm5_pinn.pt", map_location=dev)["state"])
    params = list(gnn.parameters()) + list(pinn.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    val = sample(data, 2000, dev, torch.Generator(device=dev).manual_seed(4321))
    print(f"joint training from arm5_gnn.pt + arm5_pinn.pt: {sum(p.numel() for p in params):,} parameters")

    def validate():
        gnn.eval(); pinn.eval()
        lp, lm = [], []
        for c in range(0, 2000, 250):   # chunks of 250 keep the derivative graphs within GPU memory
            b = tuple(x[c:c + 250] for x in val)
            a, m = step_losses(gnn, pinn, b, 32)
            lp.append(a.item()); lm.append(m.item())
        return sum(lp) / len(lp), sum(lm) / len(lm)

    p0, m0 = validate()
    print(f"before: pose loss {p0:.3f}  motion loss {m0:.3f}  total {p0 + W_MOTION * m0:.3f}")
    best, history, t0 = p0 + W_MOTION * m0, [], time.time()
    torch.save({"state": gnn.state_dict(), "hidden": ck["hidden"], "steps": ck["steps"]}, HERE / "arm5_gnn_joint.pt")
    torch.save({"state": pinn.state_dict()}, HERE / "arm5_pinn_joint.pt")
    for step in range(1, args.steps + 1):
        gnn.train(); pinn.train()
        lp, lm = step_losses(gnn, pinn, sample(data, args.batch, dev), args.times)
        loss = lp + W_MOTION * lm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            vp, vm = validate()
            history.append({"step": step, "pose": vp, "motion": vm, "total": vp + W_MOTION * vm})
            print(f"step {step:6d}  val pose loss {vp:.3f}  motion loss {vm:.3f}  total {vp + W_MOTION * vm:.3f}  [{time.time() - t0:4.0f}s]", flush=True)
            if vp + W_MOTION * vm < best:
                best = vp + W_MOTION * vm
                torch.save({"state": gnn.state_dict(), "hidden": ck["hidden"], "steps": ck["steps"]}, HERE / "arm5_gnn_joint.pt")
                torch.save({"state": pinn.state_dict()}, HERE / "arm5_pinn_joint.pt")
    (HERE / "joint_history.json").write_text(json.dumps(history))
    print(f"saved best pair to arm5_gnn_joint.pt + arm5_pinn_joint.pt (total {best:.3f}, started at {p0 + W_MOTION * m0:.3f})")


if __name__ == "__main__":
    main()
