"""Generate self-play games and collect (obs, policy, value, is_full) training examples."""
import random as _rng
import numpy as np
from .game import GameState, ACTION_SIZE, BOARD_SIZE
from .mcts import MCTS
from .config import Config

# ── Board symmetry augmentation ───────────────────────────────────────────────
# 8 transforms: 4 rotations × 2 reflections (dihedral group D4).
# Precomputed once at import time; policy remapping is a single fancy-index op.

def _make_perm(k: int, flip: bool, N: int = BOARD_SIZE) -> np.ndarray:
    """Return index permutation for ACTION_SIZE policy vector: new_policy = policy[perm]."""
    perm = np.zeros(N * N * 2, dtype=np.int32)
    for r in range(N):
        for c in range(N):
            nr, nc = r, c
            for _ in range(k):          # k × 90° CCW: (r,c) → (N-1-c, r)
                nr, nc = N - 1 - nc, nr
            if flip:
                nc = N - 1 - nc
            for base in (0, N * N):     # regular and strong halves
                perm[base + nr * N + nc] = base + r * N + c
    return perm


_TRANSFORMS = [(k, flip) for k in range(4) for flip in (False, True)]
_PERMS      = [_make_perm(k, flip) for k, flip in _TRANSFORMS]


def _augment(examples: list) -> list:
    """Expand examples 8× via board symmetry. Identity transform is included."""
    out = []
    for obs, policy, value, is_full in examples:
        for (k, flip), perm in zip(_TRANSFORMS, _PERMS):
            aug_obs = np.rot90(obs, k=k, axes=(1, 2))
            if flip:
                aug_obs = aug_obs[:, :, ::-1]
            out.append((aug_obs.copy(), policy[perm], value, is_full))
    return out


def play_one_game(mcts: MCTS, config: Config, opp_mcts: MCTS = None, verbose=False):
    """
    Play one game and return (examples, final_state).

    opp_mcts: if provided, one side uses this model (opponent pool). The current
              model is randomly assigned black or white; only its positions are
              collected as training examples.
    """
    state    = GameState()
    history  = []   # (obs, policy_target, player, is_full)
    captures = {1: 0, -1: 0}
    move_idx = 0

    # Opponent pool: randomly assign which player is "us"
    current_side = _rng.choice([1, -1]) if opp_mcts is not None else None

    playout_ratio = getattr(config, "PLAYOUT_CAP_RATIO", 0.0)
    fast_n        = getattr(config, "FAST_SIMULATIONS", config.NUM_SIMULATIONS)

    while not state.game_over and move_idx < config.MAX_GAME_MOVES:
        is_ours     = (current_side is None or state.current_player == current_side)
        active_mcts = mcts if is_ours else opp_mcts

        temp = 1.0 if state.move_count < config.TEMP_THRESHOLD else 0.0

        # Playout cap: apply only to our moves
        if is_ours and playout_ratio > 0:
            is_full = (_rng.random() < playout_ratio)
            n_sims  = config.NUM_SIMULATIONS if is_full else fast_n
        else:
            is_full = True
            n_sims  = None  # opponent always uses default sims

        # Temporarily override num_simulations for playout cap (C++-compatible)
        if n_sims is not None and n_sims != active_mcts.num_simulations:
            saved_n = active_mcts.num_simulations
            active_mcts.num_simulations = n_sims
            probs = active_mcts.get_action_probs(state, temperature=temp, add_noise=(temp > 0))
            active_mcts.num_simulations = saved_n
        else:
            probs = active_mcts.get_action_probs(state, temperature=temp, add_noise=(temp > 0))
        obs = state.get_observation()

        if is_ours:
            # Zero out policy for fast-search positions (not a training target)
            policy_target = probs if is_full else np.zeros(ACTION_SIZE, np.float32)
            history.append((obs, policy_target, state.current_player, is_full))

        action_idx = int(np.random.choice(ACTION_SIZE, p=probs))
        r, c, is_s = state.index_to_action(action_idx)

        if is_s and obs[2, r, c] == 1.0:
            captures[state.current_player] += 1

        state = state.make_move(r, c, is_s)
        move_idx += 1

        if verbose:
            state.render()

    if move_idx >= config.MAX_GAME_MOVES:
        state.game_over = True
        state.winner    = 0

    winner     = state.winner
    alpha      = getattr(config, "CAPTURE_REWARD_ALPHA", 0.0)
    total_caps = captures[1] + captures[-1]

    examples = []
    for obs, policy, player, is_full in history:
        z = 0.0 if winner == 0 else (1.0 if player == winner else -1.0)
        if alpha > 0.0 and total_caps > 0:
            cap_bonus = (captures[player] - captures[-player]) / total_caps
            z = (1.0 - alpha) * z + alpha * cap_bonus
        examples.append((obs, policy, np.float32(z), is_full))

    return _augment(examples), state


def generate_self_play_data(network, config: Config, num_games: int, verbose=False):
    """Run num_games self-play games. Returns flat list of (obs, policy, value, is_full)."""
    mcts = MCTS(
        network,
        num_simulations  = config.NUM_SIMULATIONS,
        c_puct           = config.C_PUCT,
        dirichlet_alpha  = config.DIRICHLET_ALPHA,
        dirichlet_eps    = config.DIRICHLET_EPS,
        leaf_batch_size  = config.LEAF_BATCH_SIZE,
        heuristic_weight = getattr(config, "HEURISTIC_WEIGHT", 0.0),
    )
    all_examples = []
    for i in range(num_games):
        mcts.clear_cache()
        examples, final = play_one_game(mcts, config, verbose=verbose)
        all_examples.extend(examples)
        w = {1: "Black", -1: "White", 0: "Draw"}.get(final.winner, "?")
        print(f"  Game {i+1}/{num_games}  "
              f"moves={final.move_count}  winner={w}  examples={len(examples)}")
    return all_examples
