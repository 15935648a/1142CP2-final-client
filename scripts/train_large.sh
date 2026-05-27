#!/usr/bin/env bash
# Large-network training loop (8 blocks, 256ch).
# Run alongside train_forever.sh in a separate tmux window.
# Launch: tmux new -s train_large 'bash scripts/train_large.sh'

set -euo pipefail
cd "$(dirname "$0")/.."

ITERS=${1:-9999}
LOG=logs/train_large.log
mkdir -p logs models

source .venv/bin/activate

echo "========================================" | tee -a "$LOG"
echo "Started: $(date)"                         | tee -a "$LOG"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), \
    '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" \
    | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

while true; do
    python main_large.py train --iterations "$ITERS" 2>&1 | tee -a "$LOG"
    EXIT=$?
    echo "$(date): exited with code $EXIT, restarting in 5s..." | tee -a "$LOG"
    sleep 5
done
