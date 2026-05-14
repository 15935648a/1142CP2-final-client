"""
AlphaZero-style MCTS with:
  - PUCT selection
  - Batched leaf evaluation (GPU-friendly)
  - Virtual loss (prevents batch sims from piling into same node)
  - Zobrist transposition cache (skip NN for seen positions)

Value convention: every node stores values from the perspective of the player
to move AT that node. UCB uses q = -child.mean_value (opponent flip).
Backprop negates value at every edge going up the tree.
"""
import math
import time
import numpy as np
from .game import GameState, ACTION_SIZE, BOARD_SIZE

EPS          = 1e-8
VIRTUAL_LOSS = 1   # virtual losses per in-flight simulation


def _threat_prior(obs: np.ndarray, legal_moves) -> np.ndarray:
    """
    Heuristic threat scores per action for cold-start bootstrapping.
    Defense (blocking opponent) is weighted 3x attack so MCTS reliably
    blocks even when the NN policy is biased toward attacking.
    Mixed into network policy via: policy * exp(weight * score).
    """
    N    = BOARD_SIZE
    own  = (obs[0] + obs[1]) > 0.5   # current player's pieces
    opp  = (obs[2] + obs[3]) > 0.5   # opponent's pieces
    scores = np.zeros(ACTION_SIZE, dtype=np.float32)
    for r, c, is_s in legal_moves:
        idx      = (N * N if is_s else 0) + r * N + c
        best_own = best_opp = 0
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cnt_own = cnt_opp = 0
            for sign in (1, -1):
                nr, nc = r + sign * dr, c + sign * dc
                while 0 <= nr < N and 0 <= nc < N and own[nr, nc]:
                    cnt_own += 1; nr += sign * dr; nc += sign * dc
            for sign in (1, -1):
                nr, nc = r + sign * dr, c + sign * dc
                while 0 <= nr < N and 0 <= nc < N and opp[nr, nc]:
                    cnt_opp += 1; nr += sign * dr; nc += sign * dc
            best_own = max(best_own, cnt_own)
            best_opp = max(best_opp, cnt_opp)
        # Defense weighted 3x: blocking a 4-row (score=12) >> extending own 4 (score=4)
        scores[idx] = best_own + 3.0 * best_opp
    return scores


class MCTSNode:
    __slots__ = ("state", "parent", "prior", "children",
                 "visit_count", "value_sum", "in_flight")

    def __init__(self, state: GameState, parent=None, prior: float = 0.0):
        self.state      = state
        self.parent     = parent
        self.prior      = prior
        self.children: dict[int, "MCTSNode"] = {}
        self.visit_count = 0
        self.value_sum   = 0.0
        self.in_flight   = 0   # active virtual losses

    @property
    def mean_value(self) -> float:
        # Each in-flight sim contributes -1 virtual loss from current player's POV
        return (self.value_sum - self.in_flight) / (self.visit_count + EPS)

    def ucb_score(self, parent_n: int, c_puct: float) -> float:
        q = -self.mean_value   # child's value from child's POV → negate for parent
        u = c_puct * self.prior * math.sqrt(parent_n) / (1.0 + self.visit_count)
        return q + u

    def best_child(self, c_puct: float):
        best_s, best_n, best_i = -math.inf, None, -1
        n = self.visit_count
        for idx, child in self.children.items():
            s = child.ucb_score(n, c_puct)
            if s > best_s:
                best_s, best_n, best_i = s, child, idx
        return best_n, best_i

    def is_leaf(self) -> bool:
        return not self.children

    def add_virtual_loss(self):
        self.in_flight   += VIRTUAL_LOSS
        self.visit_count += VIRTUAL_LOSS

    def revert_virtual_loss(self):
        self.in_flight   -= VIRTUAL_LOSS
        self.visit_count -= VIRTUAL_LOSS

    def expand(self, legal_moves, masked_priors):
        for r, c, is_s in legal_moves:
            idx        = self.state.action_to_index(r, c, is_s)
            next_state = self.state.make_move(r, c, is_s)
            self.children[idx] = MCTSNode(next_state, parent=self,
                                          prior=float(masked_priors[idx]))


class MCTS:
    def __init__(self, network, num_simulations=400, c_puct=1.5,
                 dirichlet_alpha=0.3, dirichlet_eps=0.25, leaf_batch_size=16,
                 heuristic_weight=0.0):
        self.network          = network
        self.num_simulations  = num_simulations
        self.c_puct           = c_puct
        self.dirichlet_alpha  = dirichlet_alpha
        self.dirichlet_eps    = dirichlet_eps
        self.leaf_batch_size  = leaf_batch_size
        self.heuristic_weight = heuristic_weight
        # Transposition cache: zobrist_hash → (masked_policy, value)
        self._cache: dict[int, tuple[np.ndarray, float]] = {}

    def clear_cache(self):
        self._cache.clear()

    # ------------------------------------------------------------------
    def _mask_and_normalize(self, policy: np.ndarray, legal_moves) -> np.ndarray:
        masked = np.zeros(ACTION_SIZE, dtype=np.float32)
        for r, c, is_s in legal_moves:
            idx = (BOARD_SIZE * BOARD_SIZE if is_s else 0) + r * BOARD_SIZE + c
            masked[idx] = policy[idx]
        s = masked.sum()
        if s > EPS:
            masked /= s
        else:
            w = 1.0 / len(legal_moves)
            for r, c, is_s in legal_moves:
                masked[(BOARD_SIZE * BOARD_SIZE if is_s else 0) + r * BOARD_SIZE + c] = w
        return masked

    def _ensure_cached(self, node: MCTSNode):
        """Evaluate node with NN if not in cache; expand if leaf. Returns value."""
        state = node.state
        key   = state.zobrist_hash
        if key not in self._cache:
            legal = state.get_legal_moves()
            if not legal:
                self._cache[key] = (np.zeros(ACTION_SIZE, np.float32), 0.0)
            else:
                obs    = state.get_observation()
                policy, value = self.network.predict(obs)
                if self.heuristic_weight > 0:
                    threat = _threat_prior(obs, legal)
                    policy = policy * np.exp(np.clip(threat * self.heuristic_weight, 0, 15))
                masked = self._mask_and_normalize(policy, legal)
                self._cache[key] = (masked, value)
        masked, value = self._cache[key]
        if node.is_leaf() and not state.game_over:
            legal = state.get_legal_moves()
            if legal:
                node.expand(legal, masked)
        return value

    def _add_dirichlet_noise(self, root: MCTSNode):
        children = list(root.children.values())
        if not children:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        for child, n in zip(children, noise):
            child.prior = (1 - self.dirichlet_eps) * child.prior + self.dirichlet_eps * n

    # ------------------------------------------------------------------
    def run(self, state: GameState, add_noise=True, deadline: float | None = None) -> np.ndarray:
        """
        Run MCTS and return visit-count vector of shape [ACTION_SIZE].
        If deadline is set (epoch seconds), stop when time is reached regardless of sim count.
        """
        root = MCTSNode(state)
        self._ensure_cached(root)
        if add_noise:
            self._add_dirichlet_noise(root)

        done = 0
        B    = self.leaf_batch_size

        while done < self.num_simulations and (deadline is None or time.time() < deadline):
            batch = min(B, self.num_simulations - done)

            # ── SELECT (with virtual loss) ────────────────────────────
            selected: list[tuple[MCTSNode, list[MCTSNode]]] = []
            for _ in range(batch):
                node = root
                path = [node]
                node.add_virtual_loss()
                while not node.is_leaf() and not node.state.game_over:
                    child, _ = node.best_child(self.c_puct)
                    child.add_virtual_loss()
                    node = child
                    path.append(node)
                selected.append((node, path))

            # ── BATCH EVALUATE unique non-terminal un-cached leaves ───
            to_eval: dict[int, MCTSNode] = {}   # id(node) → node
            for leaf, _ in selected:
                if leaf.state.game_over:
                    continue
                key = leaf.state.zobrist_hash
                lid = id(leaf)
                if key not in self._cache and lid not in to_eval:
                    to_eval[lid] = leaf

            if to_eval:
                nodes_list = list(to_eval.values())
                obs_arr    = np.stack([n.state.get_observation() for n in nodes_list])
                policies, values = self.network.predict_batch(obs_arr)
                for node, obs, pol, val in zip(nodes_list, obs_arr, policies, values):
                    legal = node.state.get_legal_moves()
                    if legal:
                        if self.heuristic_weight > 0:
                            threat = _threat_prior(obs, legal)
                            pol = pol * np.exp(np.clip(threat * self.heuristic_weight, 0, 15))
                        masked = self._mask_and_normalize(pol, legal)
                    else:
                        masked = np.zeros(ACTION_SIZE, np.float32)
                    self._cache[node.state.zobrist_hash] = (masked, float(val))

            # ── EXPAND + BACKPROP ─────────────────────────────────────
            for leaf, path in selected:
                # Revert virtual losses first
                for node in path:
                    node.revert_virtual_loss()

                if leaf.state.game_over:
                    value = -1.0 if leaf.state.winner != 0 else 0.0
                else:
                    key          = leaf.state.zobrist_hash
                    masked, value = self._cache[key]
                    if leaf.is_leaf():
                        legal = leaf.state.get_legal_moves()
                        if legal:
                            leaf.expand(legal, masked)

                v = value
                for node in reversed(path):
                    node.visit_count += 1
                    node.value_sum   += v
                    v = -v

            done += batch

        counts = np.zeros(ACTION_SIZE, dtype=np.float32)
        for idx, child in root.children.items():
            counts[idx] = child.visit_count
        return counts

    # ------------------------------------------------------------------
    def get_action_probs_timed(self, state: GameState, seconds: float,
                               add_noise=True) -> np.ndarray:
        """Run MCTS for up to `seconds` wall-clock seconds, then return greedy probs."""
        deadline = time.time() + seconds
        saved, self.num_simulations = self.num_simulations, 10_000_000
        try:
            counts = self.run(state, add_noise=add_noise, deadline=deadline)
        finally:
            self.num_simulations = saved
        probs = np.zeros_like(counts)
        probs[np.argmax(counts)] = 1.0
        return probs

    def get_action_probs(self, state: GameState, temperature=1.0,
                         add_noise=True) -> np.ndarray:
        counts = self.run(state, add_noise=add_noise)
        if temperature == 0:
            probs = np.zeros_like(counts)
            probs[np.argmax(counts)] = 1.0
            return probs
        counts = counts ** (1.0 / temperature)
        s = counts.sum()
        if s < EPS:
            legal = state.get_legal_moves()
            probs = np.zeros(ACTION_SIZE, dtype=np.float32)
            for r, c, is_s in legal:
                probs[state.action_to_index(r, c, is_s)] = 1.0 / len(legal)
            return probs
        return counts / s

    def select_action(self, state: GameState, temperature=1.0, add_noise=True):
        probs      = self.get_action_probs(state, temperature=temperature,
                                           add_noise=add_noise)
        action_idx = int(np.random.choice(ACTION_SIZE, p=probs))
        return state.index_to_action(action_idx), probs
