#!/usr/bin/env python3
"""Local arena: play our HeuristicAgent vs a naive RushAgent that mimics the
basic-computer opponent (rush a single line, strong-capture our blockers).

Deterministic — lets us iterate eval/search fixes with instant feedback,
validating BOTH colors before deploying to the real arena.

Usage:
    python scripts/local_arena.py            # both colors, verbose
    python scripts/local_arena.py --quiet
"""
import sys, argparse
import numpy as np
from connect6s.game import GameState, BOARD_SIZE, WIN_LENGTH
from connect6s.heuristic_agent import HeuristicAgent

N = BOARD_SIZE
DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def _board(state):
    """Signed int8 board from a GameState (black>0, white<0, |2|=strong)."""
    obs = state.get_observation()
    p = state.current_player
    b = np.zeros((N, N), dtype=np.int8)
    b += (obs[0] * p).astype(np.int8) + (obs[1] * 2 * p).astype(np.int8)
    b += (obs[2] * -p).astype(np.int8) + (obs[3] * -2 * p).astype(np.int8)
    return b


def _line_len(board, r, c, sign):
    """Longest run of `sign`-owned cells through (r,c) assuming (r,c) is owned."""
    best = 1
    for dr, dc in DIRS:
        cnt = 1
        for s in (1, -1):
            rr, cc = r + s * dr, c + s * dc
            while 0 <= rr < N and 0 <= cc < N and (int(board[rr, cc]) != 0) \
                    and (int(board[rr, cc]) > 0) == (sign > 0):
                cnt += 1
                rr += s * dr
                cc += s * dc
        best = max(best, cnt)
    return best


class RushAgent:
    """Greedy line-rusher. Approximates the arena basic computer:
    extend own longest line; use strong to capture opponent blockers or to
    place uncapturably; block opponent's near-win. Deterministic."""

    def select_action(self, state, **_):
        p = state.current_player
        board = _board(state)
        legal = state.get_legal_moves()
        best, best_score = None, -1e18

        for (r, c, is_s) in legal:
            sign = p
            # simulate placement value
            tmp = int(board[r, c])
            own_len = _line_len_with(board, r, c, sign)
            opp_block = 0
            # blocking: opponent's longest line passing adjacent to (r,c)
            opp_block = _opp_threat_at(board, r, c, -sign)
            score = own_len * 100 + opp_block * 60
            if own_len >= WIN_LENGTH:
                score += 10_000_000          # winning move
            if is_s:
                # strong: bonus when capturing opp regular (breaks their line)
                if tmp != 0 and (tmp > 0) != (sign > 0):
                    score += 250 + _line_len(board, r, c, -sign) * 40
                else:
                    score -= 30              # mild cost for spending strong on empty
            # center bias for opening
            score -= (abs(r - N // 2) + abs(c - N // 2)) * 0.5
            if score > best_score:
                best_score, best = score, (r, c, is_s)
        probs = np.zeros(450, dtype=np.float32)
        if best is None:
            best = legal[0]
        probs[state.action_to_index(*best)] = 1.0
        return best, probs


def _line_len_with(board, r, c, sign):
    old = int(board[r, c])
    board[r, c] = sign if sign > 0 else -1
    L = _line_len(board, r, c, sign)
    board[r, c] = old
    return L


def _opp_threat_at(board, r, c, opp_sign):
    """How big an opponent line would (r,c) block if placed there."""
    best = 0
    for dr, dc in DIRS:
        cnt = 0
        for s in (1, -1):
            rr, cc = r + s * dr, c + s * dc
            while 0 <= rr < N and 0 <= cc < N and int(board[rr, cc]) != 0 \
                    and (int(board[rr, cc]) > 0) == (opp_sign > 0):
                cnt += 1
                rr += s * dr
                cc += s * dc
        best = max(best, cnt)
    return best


def play_game(black_agent, white_agent, seconds=2.0, verbose=True):
    state = GameState()
    agents = {1: black_agent, -1: white_agent}
    hist = []
    while not state.game_over and state.move_count < 200:
        ag = agents[state.current_player]
        if isinstance(ag, HeuristicAgent):
            probs = ag.get_action_probs_timed(state, seconds=seconds)
            idx = int(np.argmax(probs))
            mv = state.index_to_action(idx)
        else:
            mv, _ = ag.select_action(state)
        who = "B" if state.current_player == 1 else "W"
        tag = "OUR" if isinstance(ag, HeuristicAgent) else "RUSH"
        hist.append((who, tag, mv))
        if verbose:
            print(f"  m{state.move_count:2d} {who} {tag:4s} {mv}")
        state = state.make_move(*mv)
    winner = state.winner if state.game_over else 0
    return winner, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    v = not args.quiet

    our = HeuristicAgent(depth=4)
    rush = RushAgent()

    print("=== Game 1: OUR=black vs RUSH=white ===")
    w1, _ = play_game(our, rush, args.seconds, v)
    r1 = "OUR(black) WON" if w1 == 1 else ("RUSH(white) won" if w1 == -1 else "draw")
    print("  ->", r1)

    print("=== Game 2: RUSH=black vs OUR=white ===")
    w2, _ = play_game(rush, our, args.seconds, v)
    r2 = "OUR(white) WON" if w2 == -1 else ("RUSH(black) won" if w2 == 1 else "draw")
    print("  ->", r2)

    print(f"\nSUMMARY: black-game={r1} | white-game={r2}")


if __name__ == "__main__":
    main()
