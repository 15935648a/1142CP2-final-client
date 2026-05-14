"""Generate self-play games and collect (obs, policy, value) training examples."""
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
    for obs, policy, value in examples:
        for (k, flip), perm in zip(_TRANSFORMS, _PERMS):
            aug_obs = np.rot90(obs, k=k, axes=(1, 2))
            if flip:
                aug_obs = aug_obs[:, :, ::-1]
            out.append((aug_obs.copy(), policy[perm], value))
    return out


def play_one_game(mcts: MCTS, config: Config, verbose=False):
    state   = GameState()
    history = []
    move_idx = 0

    while not state.game_over and move_idx < config.MAX_GAME_MOVES:
        if state.move_count < 8:
            temp = 3.0   # force diverse openings, prevent mode collapse
        elif state.move_count < config.TEMP_THRESHOLD:
            temp = 1.0
        else:
            temp = 0.0
        probs = mcts.get_action_probs(state, temperature=temp, add_noise=True)
        history.append((state.get_observation(), probs, state.current_player))

        action_idx = int(np.random.choice(ACTION_SIZE, p=probs))
        r, c, is_s = state.index_to_action(action_idx)
        state = state.make_move(r, c, is_s)
        move_idx += 1

        if verbose:
            state.render()

    if move_idx >= config.MAX_GAME_MOVES:
        state.game_over = True
        state.winner    = 0

    winner   = state.winner
    examples = []
    for obs, policy, player in history:
        z = 0.0 if winner == 0 else (1.0 if player == winner else -1.0)
        examples.append((obs, policy, np.float32(z)))

    return _augment(examples), state


def generate_self_play_data(network, config: Config, num_games: int, verbose=False):
    """Run num_games self-play games. Returns flat list of (obs, policy, value)."""
    mcts = MCTS(
        network,
        num_simulations  = config.NUM_SIMULATIONS,
        c_puct           = config.C_PUCT,
        dirichlet_alpha  = config.DIRICHLET_ALPHA,
        dirichlet_eps    = config.DIRICHLET_EPS,
        leaf_batch_size  = config.LEAF_BATCH_SIZE,
        heuristic_weight = getattr(config, "HEURISTIC_WEIGHT", 0.0),
    )
    MIN_MOVES = 15  # discard degenerate games to prevent mode collapse
    all_examples = []
    for i in range(num_games):
        mcts.clear_cache()   # fresh transposition table each game
        examples, final = play_one_game(mcts, config, verbose=verbose)
        all_examples.extend(examples)
        w = {1: "Black", -1: "White", 0: "Draw"}.get(final.winner, "?")
        print(f"  Game {i+1}/{num_games}  "
              f"moves={final.move_count}  winner={w}  examples={len(examples)}")
    return all_examples
