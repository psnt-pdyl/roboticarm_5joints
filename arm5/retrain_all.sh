#!/bin/sh
# Rebuild the 5-joint arm's networks after a change to the arm (motors, masses, collision rules): least-torque
# labels, the pose GNN, the motion PINN, then both trained together. Each stage starts from the previous models.
#   sh retrain_all.sh [gnn_init] [pinn_init]
set -e
cd "$(dirname "$0")"
PY="${PY:-python}"
GNN_INIT="${1:-arm5_gnn.pt}"
PINN_INIT="${2:-arm5_pinn.pt}"
echo "== labels"
"$PY" -u data.py --targets 2000000 --starts 64
echo "== pose GNN (from $GNN_INIT)"
"$PY" -u train.py --device cuda --steps 30000 --init "$GNN_INIT" --lr 5e-4 --label-weight 10 --grav-weight 0.5 --out arm5_gnn.pt
echo "== motion PINN (from $PINN_INIT)"
"$PY" -u train_pinn.py --init "$PINN_INIT" --steps 8000 --lr 3e-4 --times 32
echo "== GNN + PINN together"
"$PY" -u train_joint.py --steps 10000 --times 32
echo "== done"
