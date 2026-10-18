# Five-Joint Arm Cell

A simulated 5-joint planar robot arm that finds a part with a camera, picks it up and drops it in a
bin. Two networks drive it, both trained on the arm's physics:

- a **graph neural network (GNN)** that chooses the joint angles for a target: the pose that reaches
  it with the least motor torque, clear of the floor, the bin, the carried part and the arm itself;
- a **physics-informed neural network (PINN)** that shapes every joint move between two poses for
  low effort and low peak torque.

Run `python -m http.server 8765` in this folder and open <http://localhost:8765/>. Training code is
in `arm5/` (PyTorch).

**The arm.** Links of 220, 180, 150 and 110 mm plus a 120 mm tool (780 mm reach); joint ranges
A1 −20…200°, A2–A5 ±150°; motors of 297, 154, 111, 68 and 32 N·m; parts up to 12 kg. A target (tool
point x, z and tool angle φ) fixes only three of the five joints, so every target has infinitely
many poses. Torques come from planar recursive Newton–Euler dynamics with the part at the tool point
(`arm5/kin.py`).

## Data generation

All data is synthetic, generated from the arm's own model (`arm5/data.py`, `kin.sample_targets`).

1. **Targets.** 2,000,000 random joint poses, uniform within the joint limits, keeping every joint
   at least 20 mm above the floor. Forward kinematics turns each into a target: tool point (x, z)
   and tool angle φ. Only the target is kept, so targets cover the whole reachable workspace and
   every tool direction.
2. **Payload.** Each target gets a part mass, uniform over 0–12 kg, with a quarter at 0 kg.
3. **Label search.** For each target, 64 starting poses (63 random, one at the middle of the joint
   ranges) are each driven to the target by 30 steps of projected damped least squares
   (λ = 5, analytic Jacobian, clamped to the joint limits). An attempt is accepted if it
   - reaches within 0.5 mm and 0.1°,
   - keeps every joint 20 mm above the floor,
   - keeps every link 15 mm from the floor, the bin and the carried part,
   - keeps every part of the arm (links, joint housings, gripper) 5 mm from every other part it
     does not join, and from the pedestal.

   Among accepted attempts, the label is the one needing the **least torque to hold the payload
   still**: minimum Σᵢ (τᵢ / τ_max,ᵢ)².
4. **Result.** 1,726,426 labelled targets (86 %; the rest have no collision-free pose that was
   found). The first 20,000 are held out for validation and testing. About a minute on an RTX 4090.

The **PINN** uses no example trajectories. Each training step draws moves between labelled poses:
60 % loaded (both ends from loaded targets, payload uniform 0.5–12 kg) and 40 % empty, lasting
0.5–3 × the joint blend's duration (log-uniform), each evaluated at random moments of the move.

## GNN: the pose

`arm5/model.py`, 746,881 parameters.

**Graph.** 7 nodes: the 5 joints, the tool point and a goal node. 22 directed edges: each link in
both directions (5 × 2) and the goal to and from every other node (6 × 2).

**Inputs.**
- Node features (12): a one-hot node identity (7) and the goal vector (5), filled on the goal node
  only: `[(x − 0)/780, (z − 160)/780, sin φ, cos φ, payload/12]`.
- Edge features (5): a one-hot edge type (4: down the chain, up the chain, goal → node,
  node → goal) and the link length / 780 (0 for goal edges).

**Network.** Every MLP is Linear → SiLU → Linear → LayerNorm, hidden width 128.
1. Encode: node MLP (12 → 128), edge MLP (5 → 128).
2. Six rounds of message passing, each with its own weights:
   - edge update `e ← e + MLP([e, h_sender, h_receiver])` (384 → 128),
   - node update `h ← h + MLP([h, Σ incoming e])` (256 → 128).
3. Decode each joint node (128 → 128 → 1, no LayerNorm) to `y`, and
   `q = q_mid + q_half · tanh(y)`, so the angle always stays within the joint's range.

**Output.** The five joint angles. In the page two damped-least-squares steps then land the tool
within a fraction of a millimetre.

**Loss** (`arm5/train.py`), averaged over the batch:

```
L_pose = w_reach · [ huber(|tool point − target| / 2 mm) + huber(|tool angle − φ| / 0.5°) ]
       + w_label · Σᵢ ((qᵢ − q_label,ᵢ) / q_half,ᵢ)²                 searched least-torque pose
       + w_grav  · Σᵢ (τ_hold,ᵢ / τ_max,ᵢ)²                          torque to hold the pose still
       + 10,000  · Σᵢ [relu((qᵢ − (q_max,ᵢ − 1°)) / q_half,ᵢ)² + relu(((q_min,ᵢ + 1°) − qᵢ) / q_half,ᵢ)²]
       + 1,000   · Σ_joints relu((20 mm − z_joint) / 100 mm)²        floor
       + 1,000   · [ Σ relu((30 mm − d_cell) / 100 mm)² + Σ relu((10 mm − d_self) / 100 mm)² ]
```

`d_cell` is each link's clearance from the floor, the bin and the carried part, and `d_self` the
clearance between parts of the arm and from the pedestal. Weights: w_label = 10; w_reach ramps from
0.01 to 1 over the first 30 % of training; w_grav ramps from 0 to 0.5 between 30 % and 50 %.

**Training.** AdamW, learning rate 10⁻³ with a one-cycle schedule, batch 4,096 targets, 60,000
steps, gradients clipped at 1.0 (about 45 minutes on an RTX 4090).

## PINN: the motion

`arm5/motion.py`, 202,245 parameters.

**Form.** A rest-to-rest move from pose q0 to q1 in time T, with s = t / T:

```
q(s) = q0 + (q1 − q0) · b(s) + w(s) · N(inputs)
b(s) = 10s³ − 15s⁴ + 6s⁵        w(s) = 64 s³ (1 − s)³
```

b goes from 0 to 1, and w is 0 at both ends, so every move starts exactly at q0 and ends exactly at
q1, at rest, whatever the network outputs. The network only shapes the path in between.

**Inputs (13).** Start pose `(q0 − q_mid) / q_half` (5), pose change `(q1 − q0) / q_half` (5),
time `2s − 1`, duration `ln(T / 1 s)`, payload / 12.

**Network.** 4 layers of 256 with SiLU, then a linear layer to the 5 outputs N (initialised to
zero, so training starts from the plain blend).

**Output.** N, one correction per joint (rad). The joint angles follow from the form above; speeds
and accelerations come from the same expression by automatic differentiation, and motor torques
from Newton–Euler dynamics.

**Loss** (`arm5/train_pinn.py`), physics only, over the sampled moments of each move:

```
L_motion = mean_t Σᵢ (τᵢ / τ_max,ᵢ)²                                  effort
         + ( logsumexp(20 · |τᵢ / τ_max,ᵢ|) / 20 )²   over t and i       peak torque (soft maximum)
         + 100,000 · mean_t (joint-limit penalty, as in L_pose)
         + 1,000   · [ mean_t C(t) + max_t C(t) ]                         clearance
```

C(t) is the clearance penalty of L_pose (30 mm from the floor, the bin and the part; 10 mm between
parts of the arm). The worst moment counts as well as the mean, because a brief brush of the
gripper against the arm vanishes in a mean. There is no motor-limit term: the motor ratings were
sized from these motions (`arm5/size_motors.py`: the largest torque each joint needs over 4,000
unseen moves at 3 × the blend's duration, times 1.2, with the motors' own mass fed back until it
settles).

**Training.** AdamW, learning rate 10⁻³ with a one-cycle schedule, 512 moves per step, 24–32
moments per move.

## Train

```bash
pip install -r requirements.txt
cd arm5
python data.py --targets 2000000 --starts 64   # dataset.pt
python train.py --device cuda --steps 60000    # arm5_gnn.pt
python train_pinn.py                           # arm5_pinn.pt
python evaluate.py && python eval_pinn.py      # eval.json, pinn_eval.json
python export.py && python export_pinn.py      # ../arm5-weights.js, ../arm5-pinn-weights.js
```

## Files

| Path | What |
| --- | --- |
| `index.html` | The cell, with JavaScript ports of the networks and the dynamics |
| `arm5-weights.js`, `arm5-pinn-weights.js` | The exported GNN and PINN (generated) |
| `arm5-mgnn-weights.js` | An experimental single GNN for the whole move (`motion_gnn.py`, page mode "Motion GNN") |
| `vision-weights.js` | The camera's part detector, trained separately |
| `arm5/kin.py` | Geometry, limits, motors and masses, kinematics, dynamics, clearances |
| `arm5/data.py` | Label generation |
| `arm5/model.py`, `train.py`, `evaluate.py`, `export.py` | The GNN |
| `arm5/motion.py`, `train_pinn.py`, `eval_pinn.py`, `export_pinn.py` | The PINN |
| `arm5/*.json`, `arm5/*.xlsx` | Results and recorded inputs and outputs of the current models |
