"""Python wrapper for the C++ heuristic bot.

Implements the same interface as MCTS so it can be used as an opponent in
parallel_self_play.  Requires the connect6s_heuristic_cpp extension; build it
with:  bash scripts/build_heuristic.sh
"""
import numpy as np
from .game import GameState, ACTION_SIZE, BOARD_SIZE

N = BOARD_SIZE

try:
    import connect6s_heuristic_cpp as _cpp
    _HAVE_CPP = True
except ImportError:
    _HAVE_CPP = False


def _state_to_board(state: GameState):
    """Convert GameState to flat int8 board + metadata for C++."""
    p   = state.current_player
    obs = state.get_observation()
    flat = np.zeros(N * N, dtype=np.int8)
    for r in range(N):
        for c in range(N):
            idx = r * N + c
            if   obs[0, r, c] > 0.5: flat[idx] = p        # own regular
            elif obs[1, r, c] > 0.5: flat[idx] = p * 2    # own strong
            elif obs[2, r, c] > 0.5: flat[idx] = -p       # opp regular
            elif obs[3, r, c] > 0.5: flat[idx] = -p * 2   # opp strong
    # obs[4]=own strong count, obs[5]=opp strong count (uniform planes)
    own_cnt = int(obs[4, 0, 0] > 0.5)
    opp_cnt = int(obs[5, 0, 0] > 0.5)
    strong_b = own_cnt if p == 1 else opp_cnt
    strong_w = opp_cnt if p == 1 else own_cnt
    return flat, p, state.move_count, strong_b, strong_w


class HeuristicAgent:
    """Heuristic Connect6s bot backed by C++ alpha-beta search.

    Compatible with MCTS interface used in parallel_self_play:
      get_action_probs(state, temperature, add_noise) → np.ndarray [ACTION_SIZE]
    """

    def __init__(self, depth: int = 4):
        if not _HAVE_CPP:
            raise RuntimeError(
                "connect6s_heuristic_cpp not built — run: bash scripts/build_heuristic.sh"
            )
        self._bot  = _cpp.HeuristicBot()
        self.depth = depth
        # Attribute checked by parallel_self_play for compatibility
        self.num_simulations = 1

    def get_action_probs(self, state: GameState,
                         temperature: float = 1.0,
                         add_noise: bool = False) -> np.ndarray:
        flat, player, move_count, strong_b, strong_w = _state_to_board(state)
        idx = self._bot.best_action(flat, player, move_count,
                                    strong_b, strong_w, self.depth)
        probs = np.zeros(ACTION_SIZE, dtype=np.float32)
        probs[idx] = 1.0
        return probs

    def select_action(self, state: GameState,
                      temperature: float = 0.0,
                      add_noise: bool = False):
        probs      = self.get_action_probs(state)
        action_idx = int(np.argmax(probs))
        return state.index_to_action(action_idx), probs

    def get_action_probs_timed(self, state: GameState,
                              seconds: float = 5.0,
                              add_noise: bool = False) -> np.ndarray:
        """Iterative-deepening search within time budget."""
        flat, player, move_count, strong_b, strong_w = _state_to_board(state)
        idx = self._bot.best_action_timed(flat, player, move_count,
                                          strong_b, strong_w, float(seconds))
        probs = np.zeros(ACTION_SIZE, dtype=np.float32)
        probs[idx] = 1.0
        return probs

    def clear_cache(self):
        pass  # stateless; no-op for interface compatibility
