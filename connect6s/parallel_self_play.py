"""
Multiprocess self-play: N CPU workers generate games in parallel.
Model weights are copied to CPU before dispatch; GPU stays free for training.

Tradeoffs vs single-process:
  + Speedup ≈ min(num_workers, physical_cores)
  - Workers do CPU inference (slower per call, but parallel)
  - Process spawn overhead ~1-2s per batch
  - Model weights duplicated in RAM per worker (~50 MB each)
"""
import random as _rng
import numpy as np
import torch
import torch.multiprocessing as mp

from .network import build_net
from .mcts import MCTS
from .self_play import play_one_game
from .config import Config


# Must be a top-level function for pickle (spawn requires it)
def _worker(worker_id, state_dict, cfg_attrs, task_q, result_q):
    import os, torch as _torch
    threads = max(1, cfg_attrs.get("THREADS_PER_WORKER", 2))
    os.environ["OMP_NUM_THREADS"]      = str(threads)
    os.environ["MKL_NUM_THREADS"]      = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    _torch.set_num_threads(threads)
    _torch.set_num_interop_threads(1)

    net = build_net(cfg_attrs["NUM_RES_BLOCKS"], cfg_attrs["NUM_CHANNELS"],
                    device=torch.device("cpu"))
    net.load_state_dict(state_dict)
    net.eval()

    cfg = Config()
    for k, v in cfg_attrs.items():
        setattr(cfg, k, v)
    cfg.DEVICE = "cpu"

    mcts = MCTS(
        net,
        num_simulations  = cfg.NUM_SIMULATIONS,
        c_puct           = cfg.C_PUCT,
        dirichlet_alpha  = cfg.DIRICHLET_ALPHA,
        dirichlet_eps    = cfg.DIRICHLET_EPS,
        leaf_batch_size  = cfg.LEAF_BATCH_SIZE,
        heuristic_weight = getattr(cfg, "HEURISTIC_WEIGHT", 0.0),
    )

    # Opponent: heuristic bot (preferred) or past checkpoint pool
    opp_mcts    = None
    opp_ratio   = cfg_attrs.get("OPPONENT_POOL_RATIO", 0.0)
    use_heuristic_opp = cfg_attrs.get("HEURISTIC_OPP", False)

    if opp_ratio > 0:
        if use_heuristic_opp:
            try:
                from .heuristic_agent import HeuristicAgent
                depth    = cfg_attrs.get("HEURISTIC_OPP_DEPTH", 4)
                opp_mcts = HeuristicAgent(depth=depth)
                print(f"  [W{worker_id}] Heuristic opponent (depth={depth})", flush=True)
            except Exception as e:
                print(f"  [W{worker_id}] Heuristic opp failed: {e}", flush=True)
                opp_mcts = None
        else:
            opp_paths = cfg_attrs.get("OPPONENT_POOL_PATHS", [])
            if opp_paths:
                chosen = _rng.choice(opp_paths)
                try:
                    opp_net = build_net(cfg_attrs["NUM_RES_BLOCKS"], cfg_attrs["NUM_CHANNELS"],
                                        device=_torch.device("cpu"))
                    ckpt = _torch.load(chosen, map_location="cpu", weights_only=False)
                    sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
                    opp_net.load_state_dict(sd)
                    opp_net.eval()
                    opp_mcts = MCTS(
                        opp_net,
                        num_simulations  = cfg.NUM_SIMULATIONS,
                        c_puct           = cfg.C_PUCT,
                        dirichlet_alpha  = cfg.DIRICHLET_ALPHA,
                        dirichlet_eps    = 0.0,
                        leaf_batch_size  = cfg.LEAF_BATCH_SIZE,
                        heuristic_weight = getattr(cfg, "HEURISTIC_WEIGHT", 0.0),
                    )
                    print(f"  [W{worker_id}] Opponent pool: {chosen}", flush=True)
                except Exception as e:
                    print(f"  [W{worker_id}] Failed to load opponent {chosen}: {e}", flush=True)
                    opp_mcts = None

    all_examples = []
    games_done   = 0
    while True:
        game_idx = task_q.get()
        if game_idx is None:          # sentinel — no more work
            break
        mcts.clear_cache()
        use_opp = (opp_mcts is not None and _rng.random() < opp_ratio)
        if use_opp:
            opp_mcts.clear_cache()
        examples, final = play_one_game(
            mcts, cfg, opp_mcts=opp_mcts if use_opp else None)
        all_examples.extend(examples)
        games_done += 1
        w    = {1: "B", -1: "W", 0: "D"}.get(final.winner, "?")
        tag  = " [opp]" if use_opp else ""
        print(f"  [W{worker_id}] game={game_idx+1}  "
              f"moves={final.move_count}  winner={w}{tag}", flush=True)

    result_q.put(all_examples)


def _extract_cfg(config: Config) -> dict:
    """Pull the fields workers need (picklable primitives only)."""
    keys = [
        "NUM_RES_BLOCKS", "NUM_CHANNELS", "NUM_SIMULATIONS", "C_PUCT",
        "DIRICHLET_ALPHA", "DIRICHLET_EPS", "LEAF_BATCH_SIZE",
        "TEMP_THRESHOLD", "MAX_GAME_MOVES", "THREADS_PER_WORKER",
        "HEURISTIC_WEIGHT", "PLAYOUT_CAP_RATIO", "FAST_SIMULATIONS",
        "OPPONENT_POOL_RATIO", "CAPTURE_REWARD_ALPHA",
        "HEURISTIC_OPP", "HEURISTIC_OPP_DEPTH",
    ]
    return {k: getattr(config, k, None) for k in keys}


def generate_parallel_self_play(network, config: Config,
                                 num_games: int, num_workers: int,
                                 opponent_paths: list = None) -> list:
    """
    Run num_games self-play games across num_workers CPU processes.
    Workers pull from a shared queue — no idle waiting on stragglers.
    Returns flat list of (obs, policy, value, is_full) examples.

    opponent_paths: list of checkpoint file paths for opponent pool.
                    Workers randomly pick one at startup.
    """
    state_dict = {k: v.cpu() for k, v in network.state_dict().items()}
    cfg_attrs  = _extract_cfg(config)
    if opponent_paths:
        cfg_attrs["OPPONENT_POOL_PATHS"] = opponent_paths
    n_workers  = min(num_workers, num_games)

    ctx      = mp.get_context("spawn")
    task_q   = ctx.Queue()
    result_q = ctx.Queue()

    for i in range(num_games):
        task_q.put(i)
    for _ in range(n_workers):
        task_q.put(None)              # one sentinel per worker

    procs = [
        ctx.Process(target=_worker,
                    args=(i, state_dict, cfg_attrs, task_q, result_q))
        for i in range(n_workers)
    ]
    for p in procs:
        p.start()

    all_examples = []
    for _ in range(n_workers):
        all_examples.extend(result_q.get())

    for p in procs:
        p.join()

    full_count = sum(1 for ex in all_examples if ex[3])
    print(f"  Parallel self-play: {len(all_examples)} examples "
          f"({full_count} full-search policy targets) "
          f"from {n_workers} workers")
    return all_examples
