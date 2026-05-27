#!/usr/bin/env bash
# Quick health check before leaving training to run for weeks.
# Run: bash scripts/health_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

# Auto-detect active log: pick most recently modified between train.log and train_large.log
if [[ -z "${LOG:-}" ]]; then
    SMALL=logs/train.log
    LARGE=logs/train_large.log
    if [[ -f "$LARGE" && ( ! -f "$SMALL" || "$LARGE" -nt "$SMALL" ) ]]; then
        LOG=train_large
    else
        LOG=train
    fi
    echo "(auto-detected log: logs/${LOG}.log)"
fi

echo "=== DISK ==="
df -h . | tail -1
echo "models/ size: $(du -sh models/ 2>/dev/null | cut -f1)"

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not found)"

echo ""
echo "=== TRAINING LOG (last 30 lines) ==="
tail -30 logs/${LOG:-train}.log 2>/dev/null | cat -v | sed 's/\^\[\[[0-9;]*[mK]//g; s/\^M//g' || echo "(no log yet)"

echo ""
echo "=== LOSS TREND (value loss, last 20 epochs) ==="
grep -a "v=" logs/${LOG:-train}.log 2>/dev/null \
    | awk '{for(i=1;i<=NF;i++) if($i~/^v=/) printf "%s\n",$i}' \
    | tail -20 || echo "(no log yet)"

echo ""
echo "=== KL GAP TREND (last 10 iters, final epoch) ==="
grep -aE "Epoch ([0-9]+)/\1 " logs/${LOG:-train}.log 2>/dev/null \
    | awk '{p=""; h="";
            for(i=1;i<=NF;i++){
                if($i~/^p=/)         { sub(/^p=/,"",$i);         p=$i }
                if($i~/^H\(target\)=/){ sub(/^H\(target\)=/,"",$i); h=$i }
            }
            if(p!="" && h!="") printf "KL=%.4f  (p=%s  H=%s)\n", p-h, p, h}' \
    | tail -10 || echo "(no log yet)"

echo ""
echo "=== ARENA RESULTS (last 5) ==="
grep -a "Win-ratio" logs/${LOG:-train}.log 2>/dev/null | tail -5 || echo "(no log yet)"

echo ""
echo "=== NaN CHECK ==="
if grep -aqi "nan\|inf" logs/${LOG:-train}.log 2>/dev/null; then
    echo "WARNING: NaN/Inf detected!"
    grep -ani "nan\|inf" logs/${LOG:-train}.log | tail -5
else
    echo "No NaN/Inf. OK."
fi

echo ""
echo "=== TMUX SESSION ==="
tmux list-sessions 2>/dev/null || echo "(no tmux sessions)"
