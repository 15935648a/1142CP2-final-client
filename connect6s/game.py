import numpy as np

BOARD_SIZE = 15
WIN_LENGTH  = 6
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE * 2  # regular + strong moves

# Board cell values
EMPTY     =  0
BLACK_REG =  1
WHITE_REG = -1
BLACK_STR =  2
WHITE_STR = -2

# Strong piece award schedule: before moves 6, 13, 20, 27... (1-indexed)
# After move_count moves played, next move is move_count+1 (1-indexed)
# Award when next >= 6 and (next - 6) % 7 == 0

# ── Zobrist tables (module-level, built once) ─────────────────────────────────
# Piece types (non-empty): WHITE_STR=-2→0, WHITE_REG=-1→1, BLACK_REG=1→2, BLACK_STR=2→3
_PIECE_IDX = {-2: 0, -1: 1, 1: 2, 2: 3}

_rng = np.random.default_rng(0xDEADBEEF)
_CELL_TABLE   = _rng.integers(1, np.iinfo(np.uint64).max,
                               size=(4, BOARD_SIZE, BOARD_SIZE),
                               dtype=np.uint64)
_PLAYER_TABLE = _rng.integers(1, np.iinfo(np.uint64).max, dtype=np.uint64)
# strong count per player (0 or 1): shape [player_idx=0/1][count=0/1]
_STRONG_TABLE = _rng.integers(1, np.iinfo(np.uint64).max,
                               size=(2, 2), dtype=np.uint64)

# Initial hash for empty board, Black to move, both strong=0
_INIT_HASH = np.uint64(_STRONG_TABLE[0, 0]) ^ np.uint64(_STRONG_TABLE[1, 0])


class GameState:
    def __init__(self):
        self.board          = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = 1           # 1=Black, -1=White
        self.move_count     = 0
        self.strong_pieces  = {1: 0, -1: 0}
        self.game_over      = False
        self.winner         = None        # 1, -1, or 0 (draw)
        self.zobrist_hash   = int(_INIT_HASH)

    # ── Copy ──────────────────────────────────────────────────────────────────
    def copy(self):
        s = GameState.__new__(GameState)
        s.board          = self.board.copy()
        s.current_player = self.current_player
        s.move_count     = self.move_count
        s.strong_pieces  = self.strong_pieces.copy()
        s.game_over      = self.game_over
        s.winner         = self.winner
        s.zobrist_hash   = self.zobrist_hash
        return s

    # ── Legal moves ───────────────────────────────────────────────────────────
    def get_legal_moves(self):
        """Return list of (row, col, is_strong)."""
        p          = self.current_player
        has_strong = self.strong_pieces[p] > 0
        moves      = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cell = int(self.board[r, c])
                if cell == EMPTY:
                    moves.append((r, c, False))
                    if has_strong:
                        moves.append((r, c, True))
                elif has_strong and abs(cell) == 1:
                    moves.append((r, c, True))
        return moves

    def get_legal_move_set(self):
        return set(self.get_legal_moves())

    # ── Action encoding ───────────────────────────────────────────────────────
    def action_to_index(self, row, col, is_strong):
        flat = row * BOARD_SIZE + col
        return flat + (BOARD_SIZE * BOARD_SIZE if is_strong else 0)

    def index_to_action(self, idx):
        base = BOARD_SIZE * BOARD_SIZE
        if idx >= base:
            return (idx - base) // BOARD_SIZE, (idx - base) % BOARD_SIZE, True
        return idx // BOARD_SIZE, idx % BOARD_SIZE, False

    # ── State transition ──────────────────────────────────────────────────────
    def make_move(self, row, col, is_strong=False):
        """Return new GameState (immutable-style). Updates Zobrist hash incrementally."""
        s = self.copy()
        p    = s.current_player
        pidx = 0 if p == 1 else 1

        old_cell  = int(s.board[row, col])
        new_cell  = 2 * p if is_strong else p
        h         = s.zobrist_hash

        # --- Remove old piece from hash ---
        if old_cell != EMPTY:
            h ^= int(_CELL_TABLE[_PIECE_IDX[old_cell], row, col])

        # --- Place piece ---
        if is_strong:
            assert s.strong_pieces[p] > 0,          "no strong piece"
            assert abs(old_cell) <= 1,               "strong move must target empty or regular piece"
            # XOR out old strong count for this player
            h ^= int(_STRONG_TABLE[pidx, s.strong_pieces[p]])
            s.strong_pieces[p] -= 1
            h ^= int(_STRONG_TABLE[pidx, s.strong_pieces[p]])
        else:
            assert old_cell == EMPTY, "cell not empty"

        s.board[row, col] = np.int8(new_cell)
        h ^= int(_CELL_TABLE[_PIECE_IDX[new_cell], row, col])

        s.move_count += 1

        if s._check_win(row, col, p):
            s.game_over = True
            s.winner    = p
        elif not np.any(s.board == EMPTY):
            s.game_over = True
            s.winner    = 0

        # Toggle player
        s.current_player = -p
        h ^= int(_PLAYER_TABLE)

        # Award strong pieces if scheduled
        nxt = s.move_count + 1
        if nxt >= 6 and (nxt - 6) % 7 == 0:
            for pl, pl_idx in ((1, 0), (-1, 1)):
                old_cnt = s.strong_pieces[pl]
                if old_cnt < 1:
                    h ^= int(_STRONG_TABLE[pl_idx, old_cnt])
                    s.strong_pieces[pl] = 1
                    h ^= int(_STRONG_TABLE[pl_idx, 1])

        s.zobrist_hash = h & 0xFFFFFFFFFFFFFFFF  # keep 64-bit unsigned
        return s

    # ── Win detection ─────────────────────────────────────────────────────────
    def _check_win(self, row, col, player):
        board = self.board
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cnt = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE \
                        and board[r, c] * player > 0:
                    cnt += 1
                    r += sign * dr
                    c += sign * dc
            if cnt >= WIN_LENGTH:
                return True
        return False

    # ── Observation ───────────────────────────────────────────────────────────
    def get_observation(self):
        p   = self.current_player
        obs = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        obs[0] = (self.board ==  p).astype(np.float32)
        obs[1] = (self.board == 2*p).astype(np.float32)
        obs[2] = (self.board == -p).astype(np.float32)
        obs[3] = (self.board == -2*p).astype(np.float32)
        obs[4] = float(self.strong_pieces[p])
        obs[5] = float(self.strong_pieces[-p])
        return obs

    # ── Render ────────────────────────────────────────────────────────────────
    def render(self):
        SYM    = {0: ".", 1: "x", -1: "o", 2: "X", -2: "O"}
        header = "   " + " ".join(f"{c:2d}" for c in range(BOARD_SIZE))
        print(header)
        for r in range(BOARD_SIZE):
            row = f"{r:2d} " + " ".join(f" {SYM[int(self.board[r,c])]}"
                                         for c in range(BOARD_SIZE))
            print(row)
        pl = "Black" if self.current_player == 1 else "White"
        print(f"Move #{self.move_count + 1}  Player: {pl}  "
              f"Strong[B={self.strong_pieces[1]}, W={self.strong_pieces[-1]}]  "
              f"Hash={self.zobrist_hash:#018x}")
        if self.game_over:
            w = {1: "Black", -1: "White", 0: "Draw"}[self.winner]
            print(f"{w} wins!" if self.winner != 0 else "Draw!")


# ── C++ drop-in (overrides Python class if extension is built) ────────────────
try:
    from connect6s_cpp import GameState   # noqa: F811 — intentional override
    _USING_CPP = True
except ImportError:
    _USING_CPP = False
