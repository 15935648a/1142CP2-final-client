#!/usr/bin/env bash
# Persistent AlphaZero training loop on GPU.
# Usage: bash scripts/train_forever.sh [iterations_per_restart]
# Launch in tmux: tmux new -s train 'bash scripts/train_forever.sh'

set -euo pipefail
cd "$(dirname "$0")/.."

ITERS=${1:-9999}
LOG=logs/train.log
mkdir -p logs models

source .venv/bin/activate

echo "========================================" | tee -a "$LOG"
echo "Started: $(date)"                         | tee -a "$LOG"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), \
    '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" \
    | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

while true; do
    python main.py train --iterations "$ITERS" 2>&1 | tee -a "$LOG"
    EXIT=$?
    echo "$(date): exited with code $EXIT, restarting in 5s..." | tee -a "$LOG"
    sleep 5
done
