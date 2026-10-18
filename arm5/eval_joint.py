"""Compare the separately trained GNN + PINN pair with the jointly trained pair.

Realistic moves, as the page makes them: from a start pose, the pair's GNN chooses the end
pose for a target (refined by two damped-least-squares steps), and the pair's PINN plans
the move there. Measures the end pose (reach, holding torque) and the whole path (effort,
peak torque, collisions, closest clearance) on 1,000 unseen moves, and the vision cell's
carry to the bin with the page's duration choice. Writes joint_eval.json.
"""
import json
import math
from pathlib import Path

import torch

from eval_pinn import pick_duration, pinn_move, stats
from kin import D, TAU_MAX, dls, fk, goal_features, holding, inverse_dynamics, min_clearance
from model import ChainGNN
from motion import MotionPINN5, move_time
from train_joint import sample

HERE = Path(__file__).parent


def load_pair(gnn_file, pinn_file, dev):
    ck = torch.load(HERE / gnn_file, map_location=dev)
    gnn = ChainGNN(ck["hidden"], ck["steps"]).to(dev)
    gnn.load_state_dict(ck["state"])
    pinn = MotionPINN5().to(dev)
    pinn.load_state_dict(torch.load(HERE / pinn_file, map_location=dev)["state"])
    return gnn.eval(), pinn.eval()


def random_moves(gnn, pinn, batch, k=64, factor=1.5):
    tcp, phi, payload, _, q0, _ = batch
    with torch.no_grad():
        q1 = gnn(goal_features(tcp, phi, payload))
    q1 = dls(q1, tcp, phi, steps=2)
    reach = (fk(q1)[0] - tcp).norm(dim=-1)
    hold = (holding(q1, payload) ** 2).sum(-1)
    T = (move_time(q0, q1) * factor).clamp(min=0.15)
    s = torch.linspace(0, 1, k, device=q0.device).expand(len(q0), k)
    ex = lambda x: x[:, None].expand(-1, k, *x.shape[1:])
    q, qd, qdd = pinn.motion(ex(q0), ex(q1), s, ex(T), ex(payload))
    q, qd, qdd = q.detach(), qd.detach(), qdd.detach()
    r = inverse_dynamics(q, qd, qdd, ex(payload)) / TAU_MAX.to(q.device)
    cl = min_clearance(q, ex(payload)).min(-1).values
    return {"reach_within_1mm_pct": (reach < 1).float().mean().item() * 100, "end_holding_effort": hold.mean().item(),
            "path_effort": (r ** 2).sum(-1).mean().item(), "path_peak_pct": r.abs().flatten(-2).max(-1).values.mean().item() * 100,
            "moves_over_limit_pct": (r.abs() > 1).flatten(-2).any(-1).float().mean().item() * 100,
            "moves_colliding_pct": (cl < 0).float().mean().item() * 100, "closest_clearance_p5_mm": cl.quantile(0.05).item()}


def carry(gnn, pinn, x, kg, dev):
    P = torch.full((1,), kg, device=dev)
    def pose(px, pz):
        T, F = torch.tensor([[px, pz]], device=dev), torch.tensor([-90 * D], device=dev)
        with torch.no_grad():
            q = gnn(goal_features(T, F, P))
        return dls(q, T, F, steps=4)[0]
    q0, q1 = pose(x, 270.0), pose(-185.0, 300.0)
    T, _ = pick_duration(pinn, q0, q1, kg)
    return {**stats(*pinn_move(pinn, q0, q1, T, kg), kg), "duration_s": T}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    raw = torch.load(HERE / "dataset.pt")
    data = {k: v.to(dev) for k, v in raw.items()}
    data["n"] = len(data["q"])
    data["loaded"], data["empty"] = (data["payload"] > 0).nonzero()[:, 0], (data["payload"] == 0).nonzero()[:, 0]
    batch = sample(data, 1000, dev, torch.Generator(device=dev).manual_seed(777))
    pairs = {"separate": ("arm5_gnn.pt", "arm5_pinn.pt"), "joint": ("arm5_gnn_joint.pt", "arm5_pinn_joint.pt")}
    out = {}
    for name, files in pairs.items():
        gnn, pinn = load_pair(*files, dev)
        out[name] = {"random_moves": random_moves(gnn, pinn, batch),
                     "carry": [{"part_x": x, "kg": kg, **carry(gnn, pinn, x, kg, dev)} for kg in (2.0, 12.0) for x in (313.0, 590.0)]}
    (HERE / "joint_eval.json").write_text(json.dumps(out, indent=2))

    print("1,000 unseen moves: the pair's GNN picks the end pose, its PINN plans the move (1.5x the blend's time):")
    keys = ["reach_within_1mm_pct", "end_holding_effort", "path_effort", "path_peak_pct", "moves_over_limit_pct",
            "moves_colliding_pct", "closest_clearance_p5_mm"]
    print("  " + " " * 10 + "".join(f"{k.replace('_pct', ' %').replace('_', ' ')[:22]:>24s}" for k in keys))
    for name in pairs:
        print(f"  {name:10s}" + "".join(f"{out[name]['random_moves'][k]:24.3f}" for k in keys))
    print("\nCarry to the bin with the page's duration choice (duration / effort / peak / clearance):")
    for i, c in enumerate(out["separate"]["carry"]):
        j = out["joint"]["carry"][i]
        print(f"  part at X{c['part_x']:.0f}, {c['kg']:4.1f} kg:   separate {c['duration_s']:.1f} s / {c['effort']:.3f} / {c['peak_pct']:4.0f} % / "
              f"{c['min_clearance_mm']:5.1f} mm     joint {j['duration_s']:.1f} s / {j['effort']:.3f} / {j['peak_pct']:4.0f} % / {j['min_clearance_mm']:5.1f} mm")
    print(f"\nwrote {HERE / 'joint_eval.json'}")


if __name__ == "__main__":
    main()
