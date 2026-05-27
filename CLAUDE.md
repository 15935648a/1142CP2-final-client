# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AlphaZero-style bot for **Connect6s** (六子棋 variant) competing on Rokumoku Arena (`https://arena.ebg.tw`). 15×15 board, win = 6 in a row, with periodic **strong pieces** (awarded every 7 moves starting move 6) that can upgrade existing regular pieces.

## Commands

```bash
# Self-play bot match (watch at arena.ebg.tw/room/ROOM_ID)
bash scripts/play.sh              # time-budget mode, small model
bash scripts/play.sh 400 400      # fixed sims, small model
BOT_LARGE=1 bash scripts/play.sh  # time-budget mode, large model (best_large.pt)

# Training loop (run in tmux)
bash scripts/train_forever.sh     # small model (6b/128ch)
bash scripts/train_large.sh       # large model (8b/192ch)

# Health check
bash scripts/health_check.sh

# Arena bot (requires .env: ARENA_URL, BOT_API_KEY, ROOM_ID)
python bot.py

# One-off modes
python main.py self-play          # watch MCTS vs MCTS
python main.py play               # human vs MCTS
python main.py bench --games 20   # MCTS vs random (quick sanity check)
```

## Architecture

### Training pipeline
`main.py train` → `parallel_self_play.py` (multiprocess, CPU workers) → `trainer.py` (GPU, AMP) → arena eval → save `models/best.pt` if win-ratio ≥ 0.55.

### Key data flow
- `GameState` (immutable-style, Zobrist hash) → `MCTS.run()` → visit counts → training examples `(obs, policy, value)`
- Observation: `(6, 15, 15)` float32 — channels: own regular, own strong, opp regular, opp strong, own strong count, opp strong count
- Action space: 450 = 225 regular + 225 strong placements. Strong moves can target empty cells OR existing regular pieces.
- 8× symmetry augmentation (D4 dihedral) in `self_play.py`

### Network (`network.py`)
`PolicyValueNet`: stem Conv → N ResBlocks → policy head (→ 450 logits) + value head (→ scalar tanh). Two configs:
- Small: `NUM_RES_BLOCKS=6, NUM_CHANNELS=128` — ~6ms/eval CPU → ~3300 sims/20s
- Large: `NUM_RES_BLOCKS=8, NUM_CHANNELS=192` — ~17ms/eval CPU → ~1200 sims/20s

### MCTS (`mcts.py`)
PUCT selection, batched leaf evaluation (`LEAF_BATCH_SIZE=16`), virtual loss (parallel-safe), Zobrist transposition cache. Optional `heuristic_weight` boosts threat-blocking prior (3× defense weight) — used during self-play training, **not** in `bot.py` inference.

### Bot (`bot.py`)
Connects to arena via SSE stream. Time-budget mode (default, `BOT_SIMS=0`): spends 10–30% of remaining clock per move (scales with move number), capped at 20s. Set `BOT_SIMS>0` for fixed-sim mode.

### Configs
- `connect6s/config.py` — production (small network, 6 workers)
- `connect6s/config_large.py` — subclass for large network (3 workers, separate model paths `models/best_large.pt`)
- `main_large.py` — thin wrapper that patches `Config → ConfigLarge` before calling `main.py`

## Sandbox constraints (tournament)
CPU: 2.0, RAM: 2 GiB, Repo: 192 MiB. `bot.py` already caps `torch.set_num_threads(2)`. Only `models/best.pt` is tracked by git (`.gitignore` excludes `models/iter_*.pt`). Submit by replacing `best.pt` with the strongest checkpoint.

## Heuristic bot (`connect6s/heuristic_cpp.cpp`)
C++ alpha-beta search (depth 5, window-of-6 evaluation). Strong pieces handle all three cases: place on empty, upgrade own regular, **capture opponent regular** (breaks opponent lines). Build:
```bash
bash scripts/build_heuristic.sh
```
Python wrapper: `connect6s/heuristic_agent.py` — `HeuristicAgent(depth=4)` implements same interface as MCTS (`get_action_probs`, `select_action`).

Used as training opponent when `HEURISTIC_OPP=True` (default). 50% of self-play games are vs heuristic bot; remaining 50% are pure self-play. Current large model win rate vs heuristic: ~0% → training until model can reliably beat it.

## C++ extension
`connect6s/cpp/` compiles a drop-in `GameState` replacement. Pre-built `.so` at repo root. `game.py` auto-loads it if present (`try: from connect6s_cpp import GameState`).

## Credentials
`.env` contains `BOT_API_KEY` and `ARENA_URL`. Never commit `.env`.

## Heuristic Improvement Roadmap (deadline 2026-06-03 12:00 GMT+8)

Primary submission strategy: **BOT_HEURISTIC=1** (C++ alpha-beta beats MCTS 10/10).

### Day 1 ✅ (applied, needs build)
- `FORK_BONUS` 50k → `S_5=1,000,000` (double-4 is near-forced-win, must rank top)
- Live window multiplier `5/4` → `3/2`; dead window with n_own≥4 → `/2`
- Upgrade move scoring bug: was `def/2`, fixed to `make_score(atk/2, def)`
- Level-3 fork detection: `FORK3_BONUS = 3*S_3` when ≥2 dirs with 3-in-a-row
- Aspiration delta `S_4*3` → `S_4*5`; grow ×4 on fail (was ×2)

### Day 2
- **Defensive move guarantee**: always include moves with `def_score ≥ S_4` in candidate list, bypassing top-30 cap (prevents evicting critical blocks)
- Wire Python `find_forced_move` into C++ as pre-search check (win-in-1 / block-5 instant response)

### Day 3
- **Killer Move table**: store 2 killers per ply; try before history in move ordering
- **Null-Move Pruning**: R=2 reduction at depth≥4 when not in zugzwang (no pass move; use `--` sentinel or skip for safety at depth<4)

### Day 4
- **Incremental eval**: maintain running score updated in `Board::apply()` / undo, avoid full rescan each `evaluate()`
- OR **VCF (Victory by Consecutive Forcing)**: detect forced win sequences (5-chains) as separate search

### Day 5 (buffer)
- Arena testing: `BOT_HEURISTIC=1 bash scripts/play.sh`
- Tune `_time_budget` cap (currently 5s for heuristic) based on observed depth reached
- Final submission: confirm `best.pt` irrelevant; heuristic needs no model file
