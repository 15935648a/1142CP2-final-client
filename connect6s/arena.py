"""Pit two agents against each other to evaluate strength."""
from .game import GameState, ACTION_SIZE
import numpy as np


def play_game(agent1, agent2, verbose=False, max_moves=500):
    """
    agent1 plays Black (1), agent2 plays White (-1).
    Returns winner: 1, -1, or 0.
    """
    state = GameState()
    agents = {1: agent1, -1: agent2}
    for _ in range(max_moves):
        if state.game_over:
            break
        agent = agents[state.current_player]
        action, _ = agent.select_action(state)
        r, c, is_s = action
        state = state.make_move(r, c, is_s)
        if verbose:
            state.render()
    return state.winner


def evaluate(new_agent, old_agent, num_games=20, verbose=False):
    """
    Play num_games games alternating colors.
    Returns (new_wins, old_wins, draws, win_ratio).
    """
    new_wins = old_wins = draws = 0
    for i in range(num_games):
        if i % 2 == 0:
            # new=Black, old=White
            winner = play_game(new_agent, old_agent, verbose=verbose)
            if winner == 1:
                new_wins += 1
            elif winner == -1:
                old_wins += 1
            else:
                draws += 1
        else:
            # new=White, old=Black
            winner = play_game(old_agent, new_agent, verbose=verbose)
            if winner == -1:
                new_wins += 1
            elif winner == 1:
                old_wins += 1
            else:
                draws += 1
        print(f"  Arena game {i+1}/{num_games}: new={new_wins} old={old_wins} draws={draws}")

    total = new_wins + old_wins + draws
    win_ratio = (new_wins + 0.5 * draws) / total if total > 0 else 0.0
    return new_wins, old_wins, draws, win_ratio
