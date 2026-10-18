"""Size the five motors from the torques the trained networks' motions need.

Runs the GNN + PINN pair over the arm's working range: unseen targets at 0-12 kg, the GNN choosing each end pose and
the PINN planning the move there from another pose, at a fixed duration (by default 1.5 x the joint blend's time,
the page's usual choice), plus holding every end pose still. Each motor's rating is the largest torque its joint
needs, times a safety factor. Stronger motors make the arm heavier (kin.arm_masses: each motor sits at the joint it
drives and its mass grows with its rating), which changes the torques, so the sizing is repeated on the same motions
until it settles. Writes motor_sizing.json; prints the ratings to put in kin.TAU_MAX.

    python size_motors.py --gnn arm5_gnn_joint.pt --pinn arm5_pinn_joint.pt --factor 1.5 --safety 1.2
"""
import argparse
import json
from pathlib import Path

import torch

import kin
from kin import dls, goal_features, inverse_dynamics
from model import ChainGNN
from motion import T_MIN, MotionPINN5, move_time
from train_joint import sample

HERE = Path(__file__).parent


def use_motors(tau, dev):
    """Set kin's link masses, centres of mass and inertias for motors rated tau."""
    kin.MASS, kin.LINK_COM, kin.LINK_I = (x.to(dev) for x in kin.arm_masses(tau.cpu()))
    return kin.MASS


def motions(gnn, pinn, n, factor, dev, k=64, chunk=100):
    """Joint paths (q, qd, qdd) [n, k, 5], payloads [n], and end poses [n, 5] for n unseen moves."""
    raw = torch.load(HERE / "dataset.pt")
    data = {key: v.to(dev) for key, v in raw.items()}
    data["n"] = len(data["q"])
    data["loaded"], data["empty"] = (data["payload"] > 0).nonzero()[:, 0], (data["payload"] == 0).nonzero()[:, 0]
    tcp, phi, payload, _, q0, _ = sample(data, n, dev, torch.Generator(device=dev).manual_seed(2468))
    with torch.no_grad():
        q1 = gnn(goal_features(tcp, phi, payload))
    q1 = dls(q1, tcp, phi, steps=2).detach()
    T = (move_time(q0, q1) * factor).clamp(min=T_MIN)
    out = []
    for i in range(0, n, chunk):
        a, b, TT, pp = q0[i:i + chunk], q1[i:i + chunk], T[i:i + chunk], payload[i:i + chunk]
        s = torch.linspace(0, 1, k, device=dev).expand(len(a), k)
        ex = lambda x: x[:, None].expand(-1, k, *x.shape[1:])
        out.append(tuple(x.detach() for x in pinn.motion(ex(a), ex(b), s, ex(TT), ex(pp))))
    q, qd, qdd = (torch.cat([o[j] for o in out]) for j in range(3))
    return q, qd, qdd, payload, q1


def needed(q, qd, qdd, payload, q1):
    """|torque| per joint at each move's worst moment, and holding each end pose, with kin's current masses."""
    k = q.shape[1]
    move = torch.cat([inverse_dynamics(q[i:i + 100], qd[i:i + 100], qdd[i:i + 100], payload[i:i + 100, None].expand(-1, k)).abs().amax(1)
                      for i in range(0, len(q), 100)])
    z = torch.zeros_like(q1)
    hold = inverse_dynamics(q1, z, z, payload).abs()
    return move, hold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn", default="arm5_gnn_joint.pt")
    ap.add_argument("--pinn", default="arm5_pinn_joint.pt")
    ap.add_argument("--moves", type=int, default=4000)
    ap.add_argument("--factor", type=float, default=1.5, help="move duration, times the joint blend's")
    ap.add_argument("--safety", type=float, default=1.2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    ck = torch.load(HERE / args.gnn, map_location=dev)
    gnn = ChainGNN(ck["hidden"], ck["steps"]).to(dev).eval()
    gnn.load_state_dict(ck["state"])
    pinn = MotionPINN5().to(dev).eval()
    pinn.load_state_dict(torch.load(HERE / args.pinn, map_location=dev)["state"])
    q, qd, qdd, payload, q1 = motions(gnn, pinn, args.moves, args.factor, dev)

    tau, rounds = kin.TAU_MAX.clone(), []
    for it in range(8):                                        # rating -> link masses -> torques -> rating
        mass = use_motors(tau, dev)
        move, hold = needed(q, qd, qdd, payload, q1)
        peak = torch.maximum(move.max(0).values, hold.max(0).values).cpu()
        new = (peak * args.safety).clamp(min=1.0)
        rounds.append({"rating_N_m": tau.tolist(), "link_mass_kg": mass.cpu().tolist(), "needed_max_N_m": peak.tolist()})
        print(f"round {it + 1}: masses {[round(v, 2) for v in mass.tolist()]} kg -> needed max {[round(v, 1) for v in peak.tolist()]} N m")
        if (new - tau).abs().max() < 0.05:
            break
        tau = new
    heavy = payload >= 9
    report = {"duration_factor": args.factor, "safety": args.safety, "moves": args.moves, "rounds": rounds,
              "rating_N_m": tau.tolist(), "link_mass_kg": kin.arm_masses(tau)[0].tolist(), "motor_mass_kg": (kin.MOTOR_KG_PER_NM * tau).tolist(),
              "arm_mass_kg": float(kin.arm_masses(tau)[0].sum() + kin.MOTOR_KG_PER_NM[0] * tau[0]),
              "needed_p99_N_m": move.quantile(0.99, 0).cpu().tolist(), "needed_max_moving_N_m": move.max(0).values.cpu().tolist(),
              "needed_max_holding_N_m": hold.max(0).values.cpu().tolist(),
              "needed_max_moving_9_12kg_N_m": move[heavy].max(0).values.cpu().tolist()}
    (HERE / "motor_sizing.json").write_text(json.dumps(report, indent=2))
    print(f"\nmotors sized for these motions (max x {args.safety}):  {[round(v, 1) for v in tau.tolist()]} N m")
    print(f"link masses {[round(v, 2) for v in report['link_mass_kg']]} kg, motors {[round(v, 2) for v in report['motor_mass_kg']]} kg"
          f"  (arm {report['arm_mass_kg']:.1f} kg with the A1 motor in the pedestal)")
    print(f"needed, moving: p99 {[round(v, 1) for v in report['needed_p99_N_m']]}  max {[round(v, 1) for v in report['needed_max_moving_N_m']]} N m")
    print(f"needed, holding: max {[round(v, 1) for v in report['needed_max_holding_N_m']]} N m")
    print(f"wrote {HERE / 'motor_sizing.json'}")


if __name__ == "__main__":
    main()
