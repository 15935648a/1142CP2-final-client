"""Forced-move detection for Connect6s bot."""
import numpy as np
from .game import BOARD_SIZE, WIN_LENGTH

N = BOARD_SIZE
DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def _board_from_obs(state) -> np.ndarray:
    """Reconstruct int8 board from get_observation(). Works with C++ GameState."""
    obs = state.get_observation()
    p   = state.current_player
    b   = np.zeros((N, N), dtype=np.int8)
    b  += (obs[0] * p).astype(np.int8)
    b  += (obs[1] * (2 * p)).astype(np.int8)
    b  += (obs[2] * (-p)).astype(np.int8)
    b  += (obs[3] * (-2 * p)).astype(np.int8)
    return b


def _wins_at(board: np.ndarray, r: int, c: int, player: int, new_val: int) -> bool:
    """True if placing new_val at (r,c) gives player WIN_LENGTH in a row."""
    old = board[r, c]
    board[r, c] = new_val
    won = False
    for dr, dc in DIRS:
        cnt = 1
        for sign in (1, -1):
            rr, cc = r + sign * dr, c + sign * dc
            while 0 <= rr < N and 0 <= cc < N and int(board[rr, cc]) * player > 0:
                cnt += 1
                rr += sign * dr
                cc += sign * dc
        if cnt >= WIN_LENGTH:
            won = True
            break
    board[r, c] = old
    return won


def _find_seqs(board: np.ndarray, player: int, min_len: int) -> list:
    """All contiguous sequences of `player` with length >= min_len."""
    seqs = []
    for r in range(N):
        for c in range(N):
            if int(board[r, c]) * player <= 0:
                continue
            for dr, dc in DIRS:
                pr, pc = r - dr, c - dc
                if 0 <= pr < N and 0 <= pc < N and int(board[pr, pc]) * player > 0:
                    continue  # not the start
                cells, rr, cc = [], r, c
                while 0 <= rr < N and 0 <= cc < N and int(board[rr, cc]) * player > 0:
                    cells.append((rr, cc))
                    rr += dr
                    cc += dc
                if len(cells) >= min_len:
                    seqs.append(cells)
    return seqs


def _middle_strong_block(board: np.ndarray, opp: int,
                          tr: int, tc: int, legal_set: set):
    """
    Given threat at (tr,tc), collect opp pieces on both sides of that cell
    in each direction, then offer our strong piece at the middle of the
    longest such run. Returns (r, c, True) or None.
    """
    best, best_len = None, 0
    for dr, dc in DIRS:
        cells_bck, cells_fwd = [], []
        rr, cc = tr - dr, tc - dc
        while 0 <= rr < N and 0 <= cc < N and int(board[rr, cc]) * opp > 0:
            cells_bck.insert(0, (rr, cc))
            rr -= dr
            cc -= dc
        rr, cc = tr + dr, tc + dc
        while 0 <= rr < N and 0 <= cc < N and int(board[rr, cc]) * opp > 0:
            cells_fwd.append((rr, cc))
            rr += dr
            cc += dc
        cells = cells_bck + cells_fwd
        if len(cells) < 2 or len(cells) <= best_len:
            continue
        mr, mc = cells[len(cells) // 2]
        if (mr, mc, True) in legal_set:
            best, best_len = (mr, mc, True), len(cells)
    return best


def find_forced_move(state):
    """
    Returns (r, c, is_strong) for the highest-priority forced move, or None.

    Priority:
    1. Immediate win — prefer regular move to conserve strong piece.
    2. Block opponent's immediate winning threat:
       - If we hold a strong piece and threat is singular: place strong on
         the middle of the opponent's sequence (breaks it + captures piece).
       - Otherwise block normally.

    Proactive defense (sections 3+) removed — alpha-beta handles it.
    """
    player     = state.current_player
    opp        = -player
    has_strong = state.strong_pieces[player] > 0
    opp_has_s  = state.strong_pieces[opp] > 0

    board     = _board_from_obs(state)
    legal     = state.get_legal_moves()
    legal_set = set(legal)

    # ── 1. Immediate win ──────────────────────────────────────────────────────
    strong_win = None
    for r, c, is_s in legal:
        nv = 2 * player if is_s else player
        if _wins_at(board, r, c, player, nv):
            if not is_s:
                return (r, c, False)
            strong_win = (r, c, True)
    if strong_win:
        return strong_win

    # ── 2. Block opponent's immediate winning threats ─────────────────────────
    threats: set = set()
    for r in range(N):
        for c in range(N):
            cell = int(board[r, c])
            if cell == 0:
                if _wins_at(board, r, c, opp, opp):
                    threats.add((r, c))
            if opp_has_s and (cell == 0 or (abs(cell) == 1 and cell * opp < 0)):
                if _wins_at(board, r, c, opp, 2 * opp):
                    threats.add((r, c))

    if threats:
        if has_strong and len(threats) == 1:
            tr, tc = next(iter(threats))
            mid = _middle_strong_block(board, opp, tr, tc, legal_set)
            if mid:
                return mid
        for r, c in sorted(threats):
            if (r, c, False) in legal_set:
                return (r, c, False)
        for r, c in sorted(threats):
            if has_strong and (r, c, True) in legal_set:
                return (r, c, True)
        return None  # truly unblockable

    # ── 3b. Block fully-open opp 3-seq ───────────────────────────────────────
    # Depth-5 horizon effect: alpha-beta "plans" to block later while building
    # own live-3, but misses strong-capture danger. Force block here.
    # Skip if we already have our own live-3 — let alpha-beta extend it (fork).
    def _has_live3(p):
        for cells in _find_seqs(board, p, min_len=3):
            if len(cells) != 3:
                continue
            dr2 = cells[1][0] - cells[0][0]
            dc2 = cells[1][1] - cells[0][1]
            lo2 = (0 <= cells[0][0]-dr2 < N and 0 <= cells[0][1]-dc2 < N
                   and int(board[cells[0][0]-dr2, cells[0][1]-dc2]) == 0)
            hi2 = (0 <= cells[-1][0]+dr2 < N and 0 <= cells[-1][1]+dc2 < N
                   and int(board[cells[-1][0]+dr2, cells[-1][1]+dc2]) == 0)
            if lo2 and hi2:
                return True
        return False

    if _has_live3(player):
        return None  # we have live-3; alpha-beta will extend/fork

    seqs3 = [cells for cells in _find_seqs(board, opp, min_len=3)
             if len(cells) == 3]
    seqs3.sort(key=len, reverse=True)
    for cells in seqs3:
        dr = cells[1][0] - cells[0][0]
        dc = cells[1][1] - cells[0][1]
        lo_r, lo_c = cells[0][0] - dr, cells[0][1] - dc
        hi_r, hi_c = cells[-1][0] + dr, cells[-1][1] + dc
        lo_open = 0 <= lo_r < N and 0 <= lo_c < N and int(board[lo_r, lo_c]) == 0
        hi_open = 0 <= hi_r < N and 0 <= hi_c < N and int(board[hi_r, hi_c]) == 0
        if lo_open and hi_open:
            for r, c in [(lo_r, lo_c), (hi_r, hi_c)]:
                if (r, c, False) in legal_set:
                    return (r, c, False)
                if has_strong and (r, c, True) in legal_set:
                    return (r, c, True)

    return None
