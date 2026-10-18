"""Evaluate the 5-joint arm's GNN on unseen targets.

Reach: how close the network alone, and after two damped-least-squares steps (as the
page runs it), puts the tool point and tool angle. Torque: how hard the motors work
to hold its pose with the payload, compared with two gravity-blind ways of solving the
same targets: numerical IK started from the rest pose (50 damped-least-squares steps),
and the random pose that generated each target. Writes eval.json.
"""
import json
from pathlib import Path

import torch

from data import search
import math

from kin import D, LIM, PAYLOAD_MAX, Q_MID, SH, dls, fk, goal_features, holding, min_clearance, points, sample_targets
from model import ChainGNN

HERE = Path(__file__).parent


def reach(q, tcp, phi):
    p, a = fk(q)
    return (p - tcp).norm(dim=-1), (torch.atan2((a - phi).sin(), (a - phi).cos()) / D).abs()


def summary(q, tcp, phi, payload):
    pos, tool = reach(q, tcp, phi)
    hold = holding(q, payload).abs()
    over = (torch.relu(q - LIM[:, 1].to(q.device)) + torch.relu(LIM[:, 0].to(q.device) - q)).max(-1).values / D
    low = points(q)[..., 1:, 1].min(-1).values
    ok = pos < 1.0
    return {"tool_point_err_mm": {"mean": pos.mean().item(), "p95": pos.quantile(0.95).item(), "max": pos.max().item()},
            "tool_angle_err_deg_mean": tool.mean().item(), "within_1mm_pct": ok.float().mean().item() * 100,
            "holding_effort": (hold ** 2).sum(-1)[ok].mean().item(),
            "holding_peak_pct_mean": hold.max(-1).values[ok].mean().item() * 100,
            "over_a_motor_limit_pct": (hold.max(-1).values[ok] > 1).float().mean().item() * 100,
            "past_joint_limit_pct": (over > 0.01).float().mean().item() * 100, "max_joint_overshoot_deg": over.max().item(),
            "below_floor_pct": (low < 0).float().mean().item() * 100,
            "collides_pct": (min_clearance(q, payload) < 0).float().mean().item() * 100}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="arm5_gnn.pt")
    ap.add_argument("--out", default="eval.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(HERE / args.ckpt, map_location=dev)
    model = ChainGNN(ck["hidden"], ck["steps"]).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    tcp, phi, payload, q_gen = sample_targets(20000, dev, torch.Generator(device=dev).manual_seed(999))
    with torch.no_grad():
        q_net = model(goal_features(tcp, phi, payload))
    q_ref = dls(q_net, tcp, phi, steps=2)
    q_num = dls(Q_MID.to(dev).expand(len(tcp), -1).clone(), tcp, phi, steps=50)
    q_search = torch.cat([search(*(x[i:i + 2000] for x in (tcp, phi, payload)), 64, 30)[0] for i in range(0, len(tcp), 2000)])
    report = {"gnn_alone": summary(q_net, tcp, phi, payload), "gnn_plus_2_steps": summary(q_ref, tcp, phi, payload),
              "numerical_ik_from_rest": summary(q_num, tcp, phi, payload),
              "generating_random_pose": summary(q_gen, tcp, phi, payload),
              "least_torque_search_64": summary(q_search, tcp, phi, payload)}

    # the page's own targets: both stations, tool pointing down, heights 30-270 mm
    xs, zs = torch.tensor([300.0, 540.0]), torch.arange(30.0, 280.0, 10.0)
    T = torch.stack(torch.meshgrid(xs, zs, indexing="ij"), -1).reshape(-1, 2).to(dev)
    F = torch.full((len(T),), -90 * D, device=dev)
    report["station_targets"] = {}
    for p in (0.0, 2.0, 5.0, 12.0):
        P = torch.full((len(T),), p, device=dev)
        with torch.no_grad():
            qg = model(goal_features(T, F, P))
        qg = dls(qg, T, F, steps=2)
        qn = dls(Q_MID.to(dev).expand(len(T), -1).clone(), T, F, steps=50)
        qs = search(T, F, P, 64, 30)[0]
        eff = lambda q: (holding(q, P) ** 2).sum(-1)
        report["station_targets"][f"{p:g}_kg"] = {
            "gnn": eff(qg).mean().item(), "numerical_from_rest": eff(qn).mean().item(), "search_64": eff(qs).mean().item(),
            "gnn_beats_numerical_pct": (eff(qg) < eff(qn)).float().mean().item() * 100,
            "worst_reach_err_mm": (fk(qg)[0] - T).norm(dim=-1).max().item()}
    # the vision cell's carry route over the top (tilt out, arc, tilt in over the bin) with a 2 kg part
    R, a0, a1, rows = 640.0, 30 * D, 150 * D, []
    on_arc = lambda a: (SH[0] + R * math.cos(a), SH[1] + R * math.sin(a), a)
    lerp = lambda p, q, f: tuple(p[k] + (q[k] - p[k]) * f for k in range(3))
    for x, kg in ((200.0, 2.0), (313.0, 2.0), (450.0, 2.0), (590.0, 2.0), (313.0, 12.0), (590.0, 12.0)):
        pts = ([lerp((x, 270.0, -90 * D), on_arc(a0), i / 15) for i in range(16)] + [on_arc(a0 + (a1 - a0) * i / 15) for i in range(16)]
               + [lerp(on_arc(a1), (-185.0, 300.0, 270 * D), i / 15) for i in range(16)])
        T = torch.tensor([p[:2] for p in pts], device=dev); F = torch.tensor([p[2] for p in pts], device=dev)
        P = torch.full((len(pts),), kg, device=dev)
        with torch.no_grad():
            q = model(goal_features(T, F, P))
        q = dls(q, T, F, steps=2)
        rows.append({"part_x": x, "part_kg": kg, "min_clearance_mm": min_clearance(q, P).min().item(),
                     "worst_reach_err_mm": (fk(q)[0] - T).norm(dim=-1).max().item(),
                     "peak_holding_pct": holding(q, P).abs().max().item() * 100})
    report["carry_route"] = rows
    (HERE / args.out).write_text(json.dumps(report, indent=2))

    print(f"20,000 unseen targets, payload 0-{PAYLOAD_MAX:g} kg:")
    print(f"  {'':26s}{'tool point mm':>16s}{'within 1 mm':>13s}{'holding effort':>16s}{'peak hold':>11s}"
          f"{'over limit':>12s}{'past joint lim':>16s}{'below floor':>13s}")
    for key, label in (("gnn_alone", "GNN alone"), ("gnn_plus_2_steps", "GNN + 2 refinement steps"),
                       ("numerical_ik_from_rest", "numerical IK from rest"), ("generating_random_pose", "random generating pose"),
                       ("least_torque_search_64", "least-torque search")):
        s = report[key]
        print(f"  {label:26s}{s['tool_point_err_mm']['mean']:9.3f} ({s['tool_point_err_mm']['p95']:5.2f}){s['within_1mm_pct']:11.1f} %"
              f"{s['holding_effort']:16.3f}{s['holding_peak_pct_mean']:9.0f} %{s['over_a_motor_limit_pct']:10.1f} %"
              f"{s['past_joint_limit_pct']:9.1f} % ({s['max_joint_overshoot_deg']:4.1f}°){s['below_floor_pct']:11.1f} %")
    print("  (tool point: mean, p95 in brackets; holding figures over targets reached within 1 mm)")
    print("The page's station targets (tool down), holding effort:")
    for k, s in report["station_targets"].items():
        print(f"  {k.replace('_', ' '):6s}  GNN {s['gnn']:.3f}   numerical from rest {s['numerical_from_rest']:.3f}   "
              f"search {s['search_64']:.3f}   GNN beats numerical on {s['gnn_beats_numerical_pct']:.0f} %   "
              f"worst reach {s['worst_reach_err_mm']:.2f} mm")
    print("  collisions (random targets): " + ", ".join(f"{k} {report[k]['collides_pct']:.1f} %" for k in
          ("gnn_plus_2_steps", "numerical_ik_from_rest", "least_torque_search_64")))
    print("Carry route over the top (GNN + 2 steps at each of 48 points):")
    for r in report["carry_route"]:
        print(f"  part at X{r['part_x']:.0f}, {r['part_kg']:4.1f} kg: closest link to floor/bin/part {r['min_clearance_mm']:6.1f} mm   "
              f"worst reach {r['worst_reach_err_mm']:.2f} mm   peak holding torque {r['peak_holding_pct']:4.0f} %")
    print(f"wrote {HERE / args.out}")


if __name__ == "__main__":
    main()
