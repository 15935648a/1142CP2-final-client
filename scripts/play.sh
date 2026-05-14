#!/usr/bin/env bash
# Usage: ./scripts/play.sh [SIMS1] [SIMS2]
# 0 or omitted = time-budget mode. e.g. ./play.sh 600 400
set -euo pipefail

SIMS1=${1:-0}
SIMS2=${2:-$SIMS1}
KEY1="ra_bot_0MK0g0xYR_x1eAvwgOY9rsVLjzX0LgeN"
KEY2="ra_bot_f7GpHUwSFNNLPI5UgjqOpjyPSlEHYUiM"
BASE="https://arena.ebg.tw"
COOKIE="/tmp/arena_play_$$.txt"
LOG1="/tmp/arena_b1_$$.log"
LOG2="/tmp/arena_b2_$$.log"

cleanup() {
    echo "Stopping..."
    [ -n "${PID1:-}" ] && kill "$PID1" 2>/dev/null
    [ -n "${PID2:-}" ] && kill "$PID2" 2>/dev/null
    rm -f "$COOKIE" "$LOG1" "$LOG2"
}
trap cleanup INT TERM EXIT

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

# Login and create/reuse room
echo "Logging in..."
curl -sc "$COOKIE" -sX POST "$BASE/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"provider\":\"api_key\",\"api_key\":\"$KEY1\"}" > /dev/null

ROOM=$(curl -sb "$COOKIE" -sX POST "$BASE/api/rooms" \
  -H "Content-Type: application/json" \
  -d "{\"room_name\":\"play-$$\"}" | python3 -c "
import sys, json
d = json.load(sys.stdin)['data']
print(d.get('room_id') or d.get('active_room_id'))
")

LABEL1=$([ "$SIMS1" = "0" ] && echo "budget" || echo "${SIMS1}sims")
LABEL2=$([ "$SIMS2" = "0" ] && echo "budget" || echo "${SIMS2}sims")
echo "Room: $ROOM  (B1=$LABEL1  B2=$LABEL2)"
echo "Watch: $BASE/room/$ROOM"

# Start bots — log to files, capture real python PIDs
ARENA_URL=$BASE BOT_API_KEY=$KEY1 ROOM_ID=$ROOM BOT_SIMS=$SIMS1 python bot.py > "$LOG1" 2>&1 &
PID1=$!
sleep 0.5
ARENA_URL=$BASE BOT_API_KEY=$KEY2 ROOM_ID=$ROOM BOT_SIMS=$SIMS2 python bot.py > "$LOG2" 2>&1 &
PID2=$!

# Stream both logs with prefix
tail -f "$LOG1" | sed 's/^/[B1] /' &
tail -f "$LOG2" | sed 's/^/[B2] /' &

wait "$PID1" "$PID2"
