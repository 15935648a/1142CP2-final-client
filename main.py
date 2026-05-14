#!/usr/bin/env python3
"""
Connect-6S main entry point.

Usage:
  python main.py train             # AlphaZero self-play training loop
  python main.py self-play         # watch two MCTS agents play one game
  python main.py play              # human vs MCTS agent
  python main.py random            # two random agents (quick smoke test)
  python main.py bench             # benchmark: MCTS agent vs random
"""
import argparse
import os
import copy
import torch

from connect6s.game import GameState
from connect6s.network import build_net
from connect6s.mcts import MCTS
from connect6s.agents import RandomAgent, MCTSAgent, HumanAgent
from connect6s.self_play import generate_self_play_data
from connect6s.parallel_self_play import generate_parallel_self_play
from connect6s.trainer import Trainer
from connect6s.arena import evaluate
from connect6s.config import Config


def _print_device_info(cfg: Config):
    dev = cfg.DEVICE
    print(f"Device: {dev}")
    if "cuda" in dev:
        idx = torch.device(dev).index or 0
        print(f"  {torch.cuda.get_device_name(idx)}  "
              f"({torch.cuda.get_device_properties(idx).total_memory / 1e9:.1f} GB VRAM)")


def _load_net(cfg: Config):
    """Build net on correct device; load weights from best model if present."""
    net = build_net(cfg.NUM_RES_BLOCKS, cfg.NUM_CHANNELS, device=cfg.DEVICE)
    if os.path.exists(cfg.BEST_MODEL_PATH):
        ckpt = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        net.load_state_dict(sd)
        print(f"Loaded weights ← {cfg.BEST_MODEL_PATH}")
    return net


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_train(cfg: Config, iterations: int):
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    _print_device_info(cfg)

    net     = _load_net(cfg)
    trainer = Trainer(net, cfg)
    if os.path.exists(cfg.BEST_MODEL_PATH):
        trainer.load(cfg.BEST_MODEL_PATH, load_optimizer=True)

    EVAL_SIMS = max(cfg.NUM_SIMULATIONS // 2, 50)

    for it in range(1, iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {it}/{iterations}  {trainer.gpu_memory_str()}")
        print(f"{'='*60}")

        # 1. Self-play (parallel workers or single-process)
        print(f"\n[Self-play] {cfg.NUM_SELF_PLAY_GAMES} games × {cfg.NUM_SIMULATIONS} sims "
              f"({cfg.NUM_WORKERS} workers) ...")
        if cfg.NUM_WORKERS > 1:
            examples = generate_parallel_self_play(
                net, cfg, cfg.NUM_SELF_PLAY_GAMES, cfg.NUM_WORKERS)
        else:
            examples = generate_self_play_data(net, cfg, cfg.NUM_SELF_PLAY_GAMES)
        trainer.add_examples(examples)
        print(f"  Buffer: {len(trainer.replay)} examples  {trainer.gpu_memory_str()}")

        # 2. Train
        print(f"\n[Train] {cfg.NUM_EPOCHS} epochs ...")
        trainer.train(cfg.NUM_EPOCHS)
        trainer.step_lr_decay()

        # 3. Arena: new vs old
        print(f"\n[Arena] {cfg.NUM_EVAL_GAMES} games ...")
        old_net = build_net(cfg.NUM_RES_BLOCKS, cfg.NUM_CHANNELS, device=cfg.DEVICE)
        if os.path.exists(cfg.BEST_MODEL_PATH):
            ckpt = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE)
            sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            old_net.load_state_dict(sd)
        else:
            old_net = copy.deepcopy(net)

        new_agent = MCTSAgent(net,     num_simulations=EVAL_SIMS,
                              temperature=0.1, leaf_batch_size=cfg.LEAF_BATCH_SIZE)
        old_agent = MCTSAgent(old_net, num_simulations=EVAL_SIMS,
                              temperature=0.1, leaf_batch_size=cfg.LEAF_BATCH_SIZE)
        nw, ow, draws, ratio = evaluate(new_agent, old_agent, num_games=cfg.NUM_EVAL_GAMES)
        print(f"  New={nw}  Old={ow}  Draws={draws}  Win-ratio={ratio:.3f}")

        force = it <= cfg.FORCE_ACCEPT_ITERS
        if force or ratio >= cfg.WIN_RATIO_THRESHOLD or not os.path.exists(cfg.BEST_MODEL_PATH):
            tag = " [forced]" if force and ratio < cfg.WIN_RATIO_THRESHOLD else ""
            print(f"  ✓ Accepted (ratio={ratio:.3f}){tag}")
            trainer.save(cfg.BEST_MODEL_PATH)
        else:
            print(f"  ✗ Rejected — reloading best ...")
            trainer.load(cfg.BEST_MODEL_PATH)

        if it % cfg.CHECKPOINT_EVERY == 0:
            trainer.save(cfg.CHECKPOINT_FMT.format(it))


def cmd_self_play(cfg: Config, simulations: int):
    net = _load_net(cfg)
    mcts = MCTS(net, num_simulations=simulations, c_puct=cfg.C_PUCT,
                leaf_batch_size=cfg.LEAF_BATCH_SIZE)
    state = GameState()
    state.render()
    for _ in range(cfg.MAX_GAME_MOVES):
        if state.game_over:
            break
        action, _ = mcts.select_action(state, temperature=0.0, add_noise=False)
        r, c, is_s = action
        print(f"\n{'Black' if state.current_player==1 else 'White'} "
              f"→ ({r},{c}) {'[strong]' if is_s else ''}")
        state = state.make_move(r, c, is_s)
        state.render()
    print({1: "Black wins", -1: "White wins", 0: "Draw"}.get(state.winner, "?"))


def cmd_play(cfg: Config, simulations: int):
    net = _load_net(cfg)
    ai    = MCTSAgent(net, num_simulations=simulations, temperature=0.0,
                      leaf_batch_size=cfg.LEAF_BATCH_SIZE)
    human = HumanAgent()

    print("You=Black (x/X)  AI=White (o/O)")
    print("Input: row col   or   row col s  (strong piece)")
    state = GameState()
    state.render()

    while not state.game_over:
        if state.current_player == 1:
            action, _ = human.select_action(state)
        else:
            print("AI thinking...")
            action, _ = ai.select_action(state)
            r, c, is_s = action
            print(f"AI → ({r},{c}) {'[strong]' if is_s else ''}")
        state = state.make_move(*action)
        state.render()

    print({1: "You win!", -1: "AI wins!", 0: "Draw"}.get(state.winner, "?"))


def cmd_random():
    a1, a2 = RandomAgent(), RandomAgent()
    state  = GameState()
    for _ in range(500):
        if state.game_over:
            break
        agent  = a1 if state.current_player == 1 else a2
        action, _ = agent.select_action(state)
        state  = state.make_move(*action)
    state.render()
    w = {1: "Black", -1: "White", 0: "Draw"}.get(state.winner, "?")
    print(f"Random game: {state.move_count} moves  winner={w}")


def cmd_bench(cfg: Config, num_games: int, simulations: int):
    _print_device_info(cfg)
    net  = _load_net(cfg)
    mcts = MCTSAgent(net, num_simulations=simulations, temperature=0.0,
                     leaf_batch_size=cfg.LEAF_BATCH_SIZE)
    rand = RandomAgent()
    w, l, d, ratio = evaluate(mcts, rand, num_games=num_games)
    print(f"MCTS vs Random: W={w} L={l} D={d}  win_ratio={ratio:.3f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Connect-6S")
    sub    = parser.add_subparsers(dest="cmd")

    t = sub.add_parser("train")
    t.add_argument("--iterations", type=int, default=20)

    sp = sub.add_parser("self-play")
    sp.add_argument("--simulations", type=int, default=200)

    pl = sub.add_parser("play")
    pl.add_argument("--simulations", type=int, default=400)

    sub.add_parser("random")

    bn = sub.add_parser("bench")
    bn.add_argument("--games",       type=int, default=10)
    bn.add_argument("--simulations", type=int, default=100)

    args = parser.parse_args()
    cfg  = Config()

    dispatch = {
        "train":     lambda: cmd_train(cfg, args.iterations),
        "self-play": lambda: cmd_self_play(cfg, args.simulations),
        "play":      lambda: cmd_play(cfg, args.simulations),
        "random":    cmd_random,
        "bench":     lambda: cmd_bench(cfg, args.games, args.simulations),
    }

    if args.cmd in dispatch:
        dispatch[args.cmd]()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
