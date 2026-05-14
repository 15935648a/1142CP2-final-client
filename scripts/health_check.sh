#!/usr/bin/env bash
# Quick health check before leaving training to run for weeks.
# Run: bash scripts/health_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== DISK ==="
df -h . | tail -1
echo "models/ size: $(du -sh models/ 2>/dev/null | cut -f1)"

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not found)"

echo ""
echo "=== TRAINING LOG (last 30 lines) ==="
tail -30 logs/train.log 2>/dev/null || echo "(no log yet)"

echo ""
echo "=== LOSS TREND (value loss, last 20 epochs) ==="
grep "v=" logs/train.log 2>/dev/null \
    | awk '{for(i=1;i<=NF;i++) if($i~/^v=/) printf "%s\n",$i}' \
    | tail -20 || echo "(no log yet)"

echo ""
echo "=== ARENA RESULTS (last 5) ==="
grep "Win-ratio" logs/train.log 2>/dev/null | tail -5 || echo "(no log yet)"

echo ""
echo "=== NaN CHECK ==="
if grep -qi "nan\|inf" logs/train.log 2>/dev/null; then
    echo "WARNING: NaN/Inf detected!"
    grep -ni "nan\|inf" logs/train.log | tail -5
else
    echo "No NaN/Inf. OK."
fi

echo ""
echo "=== TMUX SESSION ==="
tmux list-sessions 2>/dev/null || echo "(no tmux sessions)"
