"""Compare the motion GNN (one network) with the pose GNN + motion PINN pair it replaces.

Same 1,000 unseen moves for both (held-out targets, start poses from other targets, 0-12 kg): the pair's end pose is
the pose GNN's, refined by two damped-least-squares steps, and the PINN plans the path; the motion GNN gives the whole
path itself, its end pose refined the same way and the correction spread over the move by b(s), as the page does.
Durations: 1.5 x the joint blend's time to the pair's end pose, for both. Also the vision cell's carry to the bin at
2 and 12 kg. Writes mgnn_eval.json.
"""
import json
import math
from pathlib import Path

import torch

from kin import D, TAU_MAX, dls, fk, goal_features, holding, inverse_dynamics, min_clearance
from model import ChainGNN
from motion import MotionPINN5, move_time
from motion_gnn import MotionGNN, blend_b
from train_mgnn import N_VAL, sample

HERE = Path(__file__).parent


def path_stats(q, qd, qdd, payload):
    k = q.shape[-2]
    P = payload[..., None].expand(*payload.shape, k)
    r = inverse_dynamics(q, qd, qdd, P) / TAU_MAX.to(q.device)
    cl = min_clearance(q, P).min(-1).values
    return {"effort": (r ** 2).sum(-1).mean(-1), "peak": r.abs().flatten(-2).max(-1).values, "min_clear": cl}


def mgnn_path(m, q0, goal, tcp, phi, T, k=64):
    """The page's motion-GNN move: network path plus the end pose's two refinement steps spread by b(s)."""
    with torch.no_grad():
        e_end = m.end_pose(q0, goal, T)
    delta = dls(e_end, tcp, phi, steps=2).detach() - e_end
    s = torch.linspace(0, 1, k, device=q0.device).expand(len(q0), k)
    with torch.no_grad():
        q, qd, qdd = m.motion(q0, goal, s, T)
    sc = s.clamp(0.005, 0.995)
    b = blend_b(sc)
    bd = 30 * sc ** 2 - 60 * sc ** 3 + 30 * sc ** 4
    bdd = 60 * sc - 180 * sc ** 2 + 120 * sc ** 3
    Tt = T[:, None, None]
    return q + b[..., None] * delta[:, None], qd + bd[..., None] * delta[:, None] / Tt, qdd + bdd[..., None] * delta[:, None] / Tt ** 2, e_end + delta


def pair_path(gnn, pinn, goal, tcp, phi, q0, payload, T, k=64):
    with torch.no_grad():
        q1 = gnn(goal)
    q1 = dls(q1, tcp, phi, steps=2).detach()
    s = torch.linspace(0, 1, k, device=q0.device).expand(len(q0), k)
    ex = lambda x: x[:, None].expand(-1, k, *x.shape[1:])
    q, qd, qdd = pinn.motion(ex(q0), ex(q1), s, ex(T), ex(payload))
    return q.detach(), qd.detach(), qdd.detach(), q1


def summary(st, q_end, tcp, payload):
    reach = (fk(q_end)[0] - tcp).norm(dim=-1)
    return {"within_1mm_pct": (reach < 1).float().mean().item() * 100, "end_holding_effort": (holding(q_end, payload) ** 2).sum(-1).mean().item(),
            "path_effort": st["effort"].mean().item(), "path_peak_pct": st["peak"].mean().item() * 100,
            "moves_over_rating_pct": (st["peak"] > 1).float().mean().item() * 100,
            "moves_colliding_pct": (st["min_clear"] < 0).float().mean().item() * 100,
            "closest_clearance_p5_mm": st["min_clear"].quantile(0.05).item()}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    raw = torch.load(HERE / "dataset.pt")
    data = {k: v.to(dev) for k, v in raw.items()}
    rows = torch.arange(N_VAL, len(data["q"]), device=dev)
    data["loaded"], data["empty"] = rows[data["payload"][N_VAL:] > 0], rows[data["payload"][N_VAL:] == 0]
    ck = torch.load(HERE / "arm5_gnn.pt", map_location=dev)
    gnn = ChainGNN(ck["hidden"], ck["steps"]).to(dev).eval()
    gnn.load_state_dict(ck["state"])
    pinn = MotionPINN5().to(dev).eval()
    pinn.load_state_dict(torch.load(HERE / "arm5_pinn.pt", map_location=dev)["state"])
    cm = torch.load(HERE / "arm5_mgnn.pt", map_location=dev)
    mg = MotionGNN(cm["hidden"], cm["steps"], cm.get("time_all", False)).to(dev).eval()
    mg.load_state_dict(cm["state"])

    tcp, phi, payload, _, q0, _ = sample(data, 1000, dev, 0, N_VAL, torch.Generator(device=dev).manual_seed(2024))
    goal = goal_features(tcp, phi, payload)
    out = {"random_moves": {"gnn_plus_pinn": {}, "motion_gnn": {}}}
    acc = {"gnn_plus_pinn": [], "motion_gnn": []}
    for c in range(0, 1000, 100):
        sl = slice(c, c + 100)
        q, qd, qdd, q1 = pair_path(gnn, pinn, goal[sl], tcp[sl], phi[sl], q0[sl], payload[sl], torch.ones(100, device=dev))
        T = (move_time(q0[sl], q1) * 1.5).clamp(min=0.15)
        q, qd, qdd, q1 = pair_path(gnn, pinn, goal[sl], tcp[sl], phi[sl], q0[sl], payload[sl], T)
        acc["gnn_plus_pinn"].append((path_stats(q, qd, qdd, payload[sl]), q1))
        q, qd, qdd, qe = mgnn_path(mg, q0[sl], goal[sl], tcp[sl], phi[sl], T)
        acc["motion_gnn"].append((path_stats(q, qd, qdd, payload[sl]), qe))
    for name, parts in acc.items():
        st = {k: torch.cat([p[0][k] for p in parts]) for k in parts[0][0]}
        out["random_moves"][name] = summary(st, torch.cat([p[1] for p in parts]), tcp, payload)

    out["carry"] = []
    for kg in (2.0, 12.0):
        for x in (313.0, 590.0):
            P = torch.full((1,), kg, device=dev)
            pose = lambda px, pz, m: dls(gnn(goal_features(torch.tensor([[px, pz]], device=dev), torch.tensor([-90 * D], device=dev), m)).detach(),
                                         torch.tensor([[px, pz]], device=dev), torch.tensor([-90 * D], device=dev), steps=4)
            q_start = pose(x, 270.0, P)
            T_bin, F_bin = torch.tensor([[-185.0, 300.0]], device=dev), torch.tensor([-90 * D], device=dev)
            g_bin = goal_features(T_bin, F_bin, P)
            row = {"part_x": x, "kg": kg}
            for f in (1.0, 1.5):
                q1 = dls(gnn(g_bin).detach(), T_bin, F_bin, steps=2).detach()
                T = (move_time(q_start, q1) * f).clamp(min=0.15)
                a = pair_path(gnn, pinn, g_bin, T_bin, F_bin, q_start, P, T, k=200)
                b = mgnn_path(mg, q_start, g_bin, T_bin, F_bin, T, k=200)
                for name, (qq, qqd, qqdd, _) in (("gnn_plus_pinn", a), ("motion_gnn", b)):
                    st = path_stats(qq, qqd, qqdd, P)
                    row[f"{name}_x{f}"] = {"duration_s": T.item(), "effort": st["effort"].item(), "peak_pct": st["peak"].item() * 100,
                                           "min_clearance_mm": st["min_clear"].item()}
            out["carry"].append(row)
    (HERE / "mgnn_eval.json").write_text(json.dumps(out, indent=2))

    keys = list(out["random_moves"]["gnn_plus_pinn"])
    print("1,000 unseen moves (0-12 kg, 1.5 x the blend's time):")
    print("  " + " " * 14 + "".join(f"{k.replace('_pct', ' %').replace('_', ' ')[:20]:>22s}" for k in keys))
    for name in ("gnn_plus_pinn", "motion_gnn"):
        print(f"  {name:14s}" + "".join(f"{out['random_moves'][name][k]:22.3f}" for k in keys))
    print("\nCarry to the bin (duration / effort / peak / closest clearance):")
    for r in out["carry"]:
        for f in (1.0, 1.5):
            a, b = r[f"gnn_plus_pinn_x{f}"], r[f"motion_gnn_x{f}"]
            print(f"  X{r['part_x']:.0f}, {r['kg']:4.1f} kg, {f}x:  GNN+PINN {a['duration_s']:.2f} s / {a['effort']:.3f} / {a['peak_pct']:4.0f} % / {a['min_clearance_mm']:6.1f} mm"
                  f"    motion GNN {b['effort']:.3f} / {b['peak_pct']:4.0f} % / {b['min_clearance_mm']:6.1f} mm")
    print(f"wrote {HERE / 'mgnn_eval.json'}")


if __name__ == "__main__":
    main()
