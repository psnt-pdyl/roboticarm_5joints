"""Evaluate the 5-joint motion PINN against the joint blend and against today's carry route.

1. 1,000 random unseen moves (0-12 kg): effort, peak torque, over-limit and collision rates.
2. The vision cell's carry, from above the part to over the bin, at 2 kg and 12 kg, four ways:
   joint blend (passes through the arm), PINN at the blend's duration, PINN with the duration
   the page picks (shortest of 1, 1.5, 2, 3 x blend time whose peak torque stays within the
   motor limits, else the lowest peak), and today's fixed over-the-top arc tracked by the GNN.
Writes pinn_eval.json.
"""
import json
import math
from pathlib import Path

import torch

from kin import D, SH, TAU_MAX, dls, dls_fast, fk, goal_features, inverse_dynamics, min_clearance
from model import ChainGNN
from motion import MotionPINN5, blend, move_time
from train_pinn import evaluate, evaluate_blend, report, sample_moves

HERE = Path(__file__).parent
FACTORS = (1.0, 1.5, 2.0, 3.0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def stats(q, qd, qdd, payload):
    r = inverse_dynamics(q, qd, qdd, torch.full(q.shape[:-1], payload, device=q.device)) / TAU_MAX.to(q.device)
    cl = min_clearance(q, torch.full(q.shape[:-1], payload, device=q.device))
    return {"effort": (r ** 2).sum(-1).mean().item(), "peak_pct": r.abs().max().item() * 100,
            "peak_per_joint_pct": (r.abs().max(0).values * 100).tolist(), "min_clearance_mm": cl.min().item()}


def pinn_move(model, q0, q1, T, payload, k=400):
    s = torch.linspace(0, 1, k, device=q0.device)
    ex = lambda x: x[None].expand(k, *x.shape)
    q, qd, qdd = model.motion(ex(q0), ex(q1), s, torch.full((k,), T, device=q0.device), torch.full((k,), payload, device=q0.device))
    return q.detach(), qd.detach(), qdd.detach()


def pick_duration(model, q0, q1, payload):
    """As the page does: the shortest candidate duration whose peak stays within the limits, else the lowest peak."""
    T0, tried = move_time(q0, q1).item(), []
    for f in FACTORS:
        st = stats(*pinn_move(model, q0, q1, T0 * f, payload, k=64), payload)
        tried.append((f, st["peak_pct"]))
        if st["peak_pct"] <= 100:
            return T0 * f, tried
    f = min(tried, key=lambda x: x[1])[0]
    return T0 * f, tried


def arc_route(gnn, x, payload, dt=1 / 240):
    """Today's carry: tilt out, arc over the top, tilt in, each a rest-to-rest move along its path,
    tracked like the page (drift toward the GNN pose every 12th step, two refinement steps)."""
    R, a0, a1 = 640.0, 30 * D, 150 * D
    on_arc = lambda a: (SH[0] + R * math.cos(a), SH[1] + R * math.sin(a), a)
    start, end = (x, 270.0, -90 * D), (-185.0, 300.0, 270 * D)
    line = lambda p, q: (lambda f: tuple(p[k] + (q[k] - p[k]) * f for k in range(3)),
                         math.hypot(q[0] - p[0], q[1] - p[1]) + 300 * abs(q[2] - p[2]))
    segs = [(line(start, on_arc(a0)), 300.0), ((lambda f: on_arc(a0 + (a1 - a0) * f), R * (a1 - a0)), 350.0),
            (line(on_arc(a1), end), 300.0)]
    dev = next(gnn.parameters()).device   # run on the CPU: thousands of tiny sequential solves
    P = torch.full((1,), payload, device=dev)
    goal = lambda p: (torch.tensor([[p[0], p[1]]], device=dev), torch.tensor([p[2]], device=dev))
    tcp, phi = goal(start)
    with torch.no_grad():
        q = dls_fast(gnn(goal_features(tcp, phi, P)), tcp, phi, steps=4)
    qs, tick, qg = [q[0]], 0, None
    for (at, L), v in segs:
        s, vel = 0.0, 0.0
        while s < L:
            vel += max(-1200 * dt, min(1200 * dt, min(v, math.sqrt(2 * 1200 * max(L - s, 0))) - vel))
            s = min(L, s + max(vel, 2) * dt)
            tcp, phi = goal(at(s / L))
            with torch.no_grad():
                if qg is None or tick % 12 == 0:
                    qg = gnn(goal_features(tcp, phi, P))
                tick += 1
                q = dls_fast(q + 0.04 * (qg - q), tcp, phi, steps=2)
            qs.append(q[0])
    q = torch.stack(qs).to(DEV)
    qd = torch.gradient(q, spacing=dt, dim=0)[0]   # finite differences of the tracked path, as the page's torque readout
    qdd = torch.gradient(qd, spacing=dt, dim=0)[0]
    return q, qd, qdd, len(qs) * dt


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MotionPINN5().to(dev)
    model.load_state_dict(torch.load(HERE / "arm5_pinn.pt", map_location=dev)["state"])
    model.eval()
    ck = torch.load(HERE / "arm5_gnn.pt", map_location=dev)
    gnn = ChainGNN(ck["hidden"], ck["steps"]).to(dev)
    gnn.load_state_dict(ck["state"])
    gnn.eval()
    gnn_cpu = ChainGNN(ck["hidden"], ck["steps"])
    gnn_cpu.load_state_dict({k: v.cpu() for k, v in ck["state"].items()})
    gnn_cpu.eval()

    raw = torch.load(HERE / "dataset.pt")
    data = {"q": raw["q"].to(dev)}
    pl = raw["payload"].to(dev)
    data["loaded"], data["empty"] = (pl > 0).nonzero()[:, 0], (pl == 0).nonzero()[:, 0]
    moves = sample_moves(data, 1000, dev, torch.Generator(device=dev).manual_seed(999))
    out = {"random_moves": {"joint_blend": report(evaluate_blend(moves)), "pinn": report(evaluate(model, moves))}}

    print("1,000 random unseen moves (0-12 kg, 0.5-3 x the blend's duration):")
    for k in ("joint_blend", "pinn"):
        s = out["random_moves"][k]
        print(f"  {k:12s} effort {s['effort']:.3f}   peak {s['peak_pct']:5.0f} %   over limit {s['over_limit_pct']:5.1f} %   "
              f"collides {s['collides_pct']:5.1f} %")

    out["carry"] = []
    print("\nThe carry from above the part to over the bin (effort / peak % of limit / closest link to part, bin, floor):")
    for kg in (2.0, 12.0):
        for x in (313.0, 590.0):
            P = torch.full((1,), kg, device=dev)
            pose = lambda px, pz: dls(gnn(goal_features(torch.tensor([[px, pz]], device=dev), torch.tensor([-90 * D], device=dev), P)).detach(),
                                      torch.tensor([[px, pz]], device=dev), torch.tensor([-90 * D], device=dev), steps=4)[0]
            q0, q1 = pose(x, 270.0), pose(-185.0, 300.0)
            T0 = move_time(q0, q1).item()
            t = torch.linspace(0, T0, 400, device=dev)
            qb = blend(q0, q1, t)
            T, tried = pick_duration(model, q0, q1, kg)
            rows = {"joint blend": (stats(*qb, kg), T0), "PINN, blend's time": (stats(*pinn_move(model, q0, q1, T0, kg), kg), T0),
                    "PINN, page's time": (stats(*pinn_move(model, q0, q1, T, kg), kg), T)}
            qa, qda, qdda, Ta = arc_route(gnn_cpu, x, kg)
            rows["arc route (today)"] = (stats(qa, qda, qdda, kg), Ta)
            out["carry"].append({"part_x": x, "kg": kg, "duration_tried": tried,
                                 **{k: {**v[0], "duration_s": v[1]} for k, v in rows.items()}})
            print(f"  part at X{x:.0f}, {kg:g} kg:")
            for k, (st, d) in rows.items():
                print(f"    {k:20s} {d:4.1f} s   effort {st['effort']:6.3f}   peak {st['peak_pct']:5.0f} %   "
                      f"clearance {st['min_clearance_mm']:6.1f} mm   per joint {[round(v) for v in st['peak_per_joint_pct']]}")
    (HERE / "pinn_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'pinn_eval.json'}")


if __name__ == "__main__":
    main()
