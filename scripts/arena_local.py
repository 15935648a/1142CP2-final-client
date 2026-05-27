#!/usr/bin/env python3
"""
Local arena: pit two MCTS configs against each other without a server.

Usage:
    python scripts/arena_local.py [--games 20] [--budget 5.0] [--sims 400]

  --budget   think seconds for "budget" player  (default 5.0)
  --sims     fixed sim count for "fixed" player (default 400)
  --games    total games to play                (default 20)

Players alternate colors each game.
"""
import argparse
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from connect6s.config import Config
from connect6s.config_large import ConfigLarge
from connect6s.game import GameState
from connect6s.heuristic_agent import HeuristicAgent
from connect6s.mcts import MCTS
from connect6s.network import build_net

torch.set_num_threads(2)
torch.set_num_interop_threads(1)


def build_mcts(net, cfg, num_simulations):
    return MCTS(
        net,
        num_simulations = num_simulations,
        c_puct          = cfg.C_PUCT,
        dirichlet_alpha = cfg.DIRICHLET_ALPHA,
        dirichlet_eps   = 0.0,
        leaf_batch_size = cfg.LEAF_BATCH_SIZE,
    )


def play_game(mcts_black, mcts_white, budget_black, budget_white, max_moves=500):
    """Play one game. Returns winner: 1=black, 2=white, 0=draw."""
    state = GameState()
    mcts_map  = {1: mcts_black, 2: mcts_white}
    budget_map = {1: budget_black, 2: budget_white}
    move_no = 0

    while not state.game_over and move_no < max_moves:
        cp      = state.current_player          # 1=black, -1=white
        player  = 1 if cp == 1 else 2
        mcts    = mcts_map[player]
        budget  = budget_map[player]

        mcts.clear_cache()
        if budget is not None:
            probs = mcts.get_action_probs_timed(state, seconds=budget, add_noise=False)
        else:
            probs = mcts.get_action_probs(state, temperature=0.0, add_noise=False)

        idx         = int(np.argmax(probs))
        row, col, s = state.index_to_action(idx)
        state       = state.make_move(row, col, s)
        move_no    += 1

    if state.game_over:
        return state.winner   # 1 or 2
    return 0                  # draw / timeout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games",     type=int,   default=20)
    parser.add_argument("--budget",    type=float, default=5.0,
                        help="Think seconds for MCTS budget player")
    parser.add_argument("--sims",      type=int,   default=400,
                        help="Fixed sim count for MCTS fixed player")
    parser.add_argument("--heuristic", action="store_true",
                        help="Replace 'fixed' player with HeuristicAgent(depth=4)")
    parser.add_argument("--large",     action="store_true",
                        help="Use large model (best_large.pt)")
    args = parser.parse_args()

    cfg    = ConfigLarge() if args.large else Config()
    device = torch.device("cpu")
    net    = build_net(cfg.NUM_RES_BLOCKS, cfg.NUM_CHANNELS, device=device)

    ckpt = torch.load(cfg.BEST_MODEL_PATH, map_location=device)
    sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(sd)
    net.eval()
    print(f"Loaded {cfg.BEST_MODEL_PATH}")

    mcts_budget = build_mcts(net, cfg, num_simulations=10_000_000)

    if args.heuristic:
        challenger    = HeuristicAgent(depth=4)
        p2_name       = "heuristic(d4)"
        p2_budget     = None   # heuristic ignores budget
        print(f"\nMCTS(budget={args.budget}s) vs Heuristic(depth=4)  —  {args.games} games\n")
    else:
        challenger    = build_mcts(net, cfg, num_simulations=args.sims)
        p2_name       = f"fixed({args.sims})"
        p2_budget     = None
        print(f"\nBudget({args.budget}s) vs Fixed({args.sims}sims)  —  {args.games} games\n")

    wins = {"budget": 0, "challenger": 0, "draw": 0}

    for g in range(args.games):
        if g % 2 == 0:
            b_player, w_player = "budget", "challenger"
            mb, mw = mcts_budget, challenger
            bb, bw = args.budget, p2_budget
        else:
            b_player, w_player = "challenger", "budget"
            mb, mw = challenger, mcts_budget
            bb, bw = p2_budget, args.budget

        t0      = time.time()
        winner  = play_game(mb, mw, bb, bw)
        elapsed = time.time() - t0

        if winner == 1:
            wins[b_player] += 1
            result = f"Black({b_player}) wins"
        elif winner == -1:
            wins[w_player] += 1
            result = f"White({w_player}) wins"
        else:
            wins["draw"] += 1
            result = "Draw"

        print(f"Game {g+1:3d}/{args.games}  {result:40s}  {elapsed:.0f}s")

    total = args.games - wins["draw"]
    print(f"\n{'='*50}")
    print(f"budget({args.budget}s):  {wins['budget']} wins")
    print(f"{p2_name}: {wins['challenger']} wins")
    print(f"draws:           {wins['draw']}")
    if total > 0:
        print(f"MCTS win rate:       {wins['budget']/total*100:.1f}% (excl. draws)")
        print(f"Challenger win rate: {wins['challenger']/total*100:.1f}% (excl. draws)")


if __name__ == "__main__":
    main()
