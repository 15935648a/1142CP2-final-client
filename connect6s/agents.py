"""Agent interfaces: Random, MCTS-based, Human."""
import random
import numpy as np
from .game import GameState, ACTION_SIZE
from .mcts import MCTS


class RandomAgent:
    def select_action(self, state: GameState):
        moves = state.get_legal_moves()
        return random.choice(moves), None


class MCTSAgent:
    def __init__(self, network, num_simulations=200, temperature=0.0,
                 c_puct=1.5, leaf_batch_size=16,
                 dirichlet_alpha=0.03, dirichlet_eps=0.25):
        self.mcts = MCTS(network, num_simulations=num_simulations,
                         c_puct=c_puct, leaf_batch_size=leaf_batch_size,
                         dirichlet_alpha=dirichlet_alpha,
                         dirichlet_eps=dirichlet_eps)
        self.temperature = temperature

    def select_action(self, state: GameState):
        action, probs = self.mcts.select_action(
            state, temperature=self.temperature, add_noise=False
        )
        return action, probs


class HumanAgent:
    """CLI human player."""
    def select_action(self, state: GameState):
        legal = state.get_legal_move_set()
        while True:
            try:
                raw = input("Your move (row col [s for strong]): ").strip().split()
                r, c = int(raw[0]), int(raw[1])
                is_s = len(raw) > 2 and raw[2].lower() == "s"
                if (r, c, is_s) in legal:
                    return (r, c, is_s), None
                print("Illegal move.")
            except (ValueError, IndexError):
                print("Format: row col [s]")
