#!/usr/bin/env python3
"""Parse bot_test.log into structured JSON per game."""
import json, re, sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bot_test.log"

# Patterns
RE_GAME_OVER = re.compile(r"\[GAME OVER\] winner=(\S+) I_am=(\S+) moves=(\d+)")
RE_HIST      = re.compile(r"\[HIST\] move=(\d+) (ME  |OPP ) \((\d+),(\d+)\) strong=(True|False)")
RE_MOVED     = re.compile(r"Moved \((\d+),(\d+)\) strong=(True|False)")
RE_FORCED    = re.compile(r"Forced move: \((\d+),(\d+)\) strong=(True|False)")
RE_THINKING  = re.compile(r"Thinking for ([\d.]+)s \(bank=([\d.]+)s move=(\d+)\)")
RE_MCTS      = re.compile(r"MCTS done in ([\d.]+)s")
RE_BID       = re.compile(r"Bid value=([\d.]+) color=(\S+)")
RE_SEAT      = re.compile(r"Requesting seat (\S+)")
RE_MODE      = re.compile(r"Mode: (.+)")
RE_TIME      = re.compile(r"^(\d+:\d+:\d+)")

games = []
current = None

with open(LOG) as f:
    for line in f:
        line = line.strip()
        ts_m = RE_TIME.match(line)
        ts = ts_m.group(1) if ts_m else ""

        m = RE_GAME_OVER.search(line)
        if m:
            if current:
                current["result"] = {"winner": m.group(1), "i_am": m.group(2), "total_moves": int(m.group(3))}
                games.append(current)
            current = {"result": None, "history": [], "my_moves": [], "think_times": []}
            continue

        if current is None:
            current = {"result": None, "history": [], "my_moves": [], "think_times": []}

        m = RE_HIST.search(line)
        if m:
            current["history"].append({
                "move": int(m.group(1)), "who": m.group(2).strip(),
                "r": int(m.group(3)), "c": int(m.group(4)), "strong": m.group(5) == "True"
            })
            continue

        m = RE_FORCED.search(line)
        if m:
            current["my_moves"].append({"ts": ts, "r": int(m.group(1)), "c": int(m.group(2)),
                                         "strong": m.group(3) == "True", "forced": True})
            continue

        m = RE_MOVED.search(line)
        if m:
            entry = {"ts": ts, "r": int(m.group(1)), "c": int(m.group(2)),
                     "strong": m.group(3) == "True", "forced": False}
            # merge with last forced if same coords
            if current["my_moves"] and current["my_moves"][-1].get("forced"):
                last = current["my_moves"][-1]
                if last["r"] == entry["r"] and last["c"] == entry["c"]:
                    continue
            current["my_moves"].append(entry)
            continue

        m = RE_THINKING.search(line)
        if m:
            current["think_times"].append({"ts": ts, "budget": float(m.group(1)),
                                            "bank": float(m.group(2)), "move_no": int(m.group(3))})
            continue

        m = RE_BID.search(line)
        if m:
            current["bid"] = {"value": float(m.group(1)), "color": m.group(2)}
            continue

        m = RE_SEAT.search(line)
        if m:
            current.setdefault("seat_requests", []).append(m.group(1))
            continue

        m = RE_MODE.search(line)
        if m:
            current["mode"] = m.group(1)
            continue

if current and current.get("result"):
    games.append(current)
elif current:
    current["result"] = {"winner": "?", "i_am": "?", "total_moves": "?", "note": "game incomplete"}
    games.append(current)

for i, g in enumerate(games):
    g["game_number"] = i + 1

print(json.dumps(games, indent=2))
