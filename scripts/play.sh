#!/usr/bin/env bash
# Usage: ./scripts/play.sh [SIMS]
# Runs both bots in a new Arena room. Ctrl+C to stop both.
set -euo pipefail

SIMS1=${1:-0}   # 0 = use time budget
SIMS2=${2:-$SIMS1}
KEY1="ra_bot_0MK0g0xYR_x1eAvwgOY9rsVLjzX0LgeN"
KEY2="ra_bot_f7GpHUwSFNNLPI5UgjqOpjyPSlEHYUiM"
BASE="https://arena.ebg.tw"
COOKIE="/tmp/arena_play_$$.txt"

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

# Login with bot1
echo "Logging in..."
curl -sc "$COOKIE" -sX POST "$BASE/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"provider\":\"api_key\",\"api_key\":\"$KEY1\"}" > /dev/null

# Create room; if blocked by active room, reuse that room
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

# Start both bots
ARENA_URL=$BASE BOT_API_KEY=$KEY1 ROOM_ID=$ROOM BOT_SIMS=$SIMS1 python bot.py 2>&1 | sed 's/^/[B1] /' &
PID1=$!
sleep 0.5
ARENA_URL=$BASE BOT_API_KEY=$KEY2 ROOM_ID=$ROOM BOT_SIMS=$SIMS2 python bot.py 2>&1 | sed 's/^/[B2] /' &
PID2=$!

trap "echo 'Stopping...'; kill $PID1 $PID2 2>/dev/null; rm -f $COOKIE" INT TERM
wait $PID1 $PID2
rm -f "$COOKIE"
