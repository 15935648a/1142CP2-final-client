"""
Multiprocess self-play: N CPU workers generate games in parallel.
Model weights are copied to CPU before dispatch; GPU stays free for training.

Tradeoffs vs single-process:
  + Speedup ≈ min(num_workers, physical_cores)
  - Workers do CPU inference (slower per call, but parallel)
  - Process spawn overhead ~1-2s per batch
  - Model weights duplicated in RAM per worker (~50 MB each)
"""
import numpy as np
import torch
import torch.multiprocessing as mp

from .network import build_net
from .mcts import MCTS
from .self_play import play_one_game
from .config import Config


# Must be a top-level function for pickle (spawn requires it)
def _worker(args):
    worker_id, state_dict, cfg_attrs, num_games = args

    # Prevent thread explosion: each worker gets its fair share of cores.
    # Without this, N workers × torch_default_threads thrash all CPU cores.
    import os, torch as _torch
    threads = max(1, cfg_attrs.get("THREADS_PER_WORKER", 2))
    os.environ["OMP_NUM_THREADS"]     = str(threads)
    os.environ["MKL_NUM_THREADS"]     = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    _torch.set_num_threads(threads)
    _torch.set_num_interop_threads(1)

    net = build_net(cfg_attrs["NUM_RES_BLOCKS"], cfg_attrs["NUM_CHANNELS"],
                    device=torch.device("cpu"))
    net.load_state_dict(state_dict)
    net.eval()

    # Rebuild minimal config for this worker
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

    MIN_MOVES = 15
    all_examples = []
    for i in range(num_games):
        mcts.clear_cache()
        examples, final = play_one_game(mcts, cfg)
        all_examples.extend(examples)
        w = {1: "B", -1: "W", 0: "D"}.get(final.winner, "?")
        print(f"  [W{worker_id}] {i+1}/{num_games}  "
              f"moves={final.move_count}  winner={w}", flush=True)

    return all_examples


def _extract_cfg(config: Config) -> dict:
    """Pull the fields workers need (picklable primitives only)."""
    keys = [
        "NUM_RES_BLOCKS", "NUM_CHANNELS", "NUM_SIMULATIONS", "C_PUCT",
        "DIRICHLET_ALPHA", "DIRICHLET_EPS", "LEAF_BATCH_SIZE",
        "TEMP_THRESHOLD", "MAX_GAME_MOVES", "THREADS_PER_WORKER",
        "HEURISTIC_WEIGHT",
    ]
    return {k: getattr(config, k) for k in keys}


def generate_parallel_self_play(network, config: Config,
                                 num_games: int, num_workers: int) -> list:
    """
    Run num_games self-play games across num_workers CPU processes.
    Returns flat list of (obs, policy, value) examples.
    """
    # Copy weights to CPU tensors (GPU tensors are not fork-safe)
    state_dict = {k: v.cpu() for k, v in network.state_dict().items()}
    cfg_attrs  = _extract_cfg(config)

    base  = num_games // num_workers
    extra = num_games  % num_workers
    split = [base + (1 if i < extra else 0) for i in range(num_workers)]
    split = [g for g in split if g > 0]   # drop zero-game slots

    worker_args = [(i, state_dict, cfg_attrs, g) for i, g in enumerate(split)]

    ctx = mp.get_context("spawn")
    with ctx.Pool(len(worker_args)) as pool:
        results = pool.map(_worker, worker_args)

    all_examples = [ex for batch in results for ex in batch]
    print(f"  Parallel self-play: {len(all_examples)} examples "
          f"from {len(worker_args)} workers")
    return all_examples
