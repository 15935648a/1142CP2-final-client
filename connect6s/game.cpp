#include "game.hpp"
#include <cassert>
#include <cstdio>
#include <random>

// ── Zobrist table initialisation ──────────────────────────────────────────────
static ZobristTables build_zobrist() {
    std::mt19937_64 rng(0xDEADBEEFCAFE0000ULL);
    ZobristTables z{};
    for (int p = 0; p < 4; ++p)
        for (int r = 0; r < BOARD_SIZE; ++r)
            for (int c = 0; c < BOARD_SIZE; ++c)
                z.cell[p][r][c] = rng();
    z.player = rng();
    for (int pi = 0; pi < 2; ++pi)
        for (int cnt = 0; cnt < 2; ++cnt)
            z.strong[pi][cnt] = rng();
    // Empty board, Black to move: XOR in "0 strong" tokens for both players
    z.init_hash = z.strong[0][0] ^ z.strong[1][0];
    return z;
}

const ZobristTables ZOBRIST = build_zobrist();

// ── GameState::clear ──────────────────────────────────────────────────────────
void GameState::clear() {
    for (auto& row : board) row.fill(EMPTY);
    current_player   = 1;
    move_count       = 0;
    strong_pieces[0] = strong_pieces[1] = 0;
    game_over        = false;
    winner           = 0;
    zobrist_hash     = ZOBRIST.init_hash;
}

// ── GameState::legal_moves ────────────────────────────────────────────────────
std::vector<Move> GameState::legal_moves() const {
    std::vector<Move> moves;
    moves.reserve(BOARD_SIZE * BOARD_SIZE * 2);
    bool has_strong = strong_for(current_player) > 0;

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int8_t v = board[r][c];
            if (v == EMPTY) {
                moves.push_back({r, c, false});
                if (has_strong) moves.push_back({r, c, true});
            } else if (has_strong && std::abs(static_cast<int>(v)) == 1) {
                moves.push_back({r, c, true});
            }
        }
    }
    return moves;
}

// ── GameState::apply ──────────────────────────────────────────────────────────
GameState GameState::apply(Move mv) const {
    GameState s = *this;
    int     p    = s.current_player;
    int     pidx = player_idx(p);
    int8_t  old  = s.board[mv.row][mv.col];
    int8_t  nw   = static_cast<int8_t>(mv.is_strong ? 2 * p : p);
    uint64_t& h  = s.zobrist_hash;

    // Remove old piece from hash
    if (old != EMPTY)
        h ^= ZOBRIST.cell[piece_idx(old)][mv.row][mv.col];

    // Place piece
    if (mv.is_strong) {
        assert(s.strong_pieces[pidx] > 0);
        assert(old == static_cast<int8_t>(-p));
        int old_cnt = s.strong_pieces[pidx];
        h ^= ZOBRIST.strong[pidx][old_cnt];
        s.strong_pieces[pidx]--;
        h ^= ZOBRIST.strong[pidx][s.strong_pieces[pidx]];
    } else {
        assert(old == EMPTY);
    }

    s.board[mv.row][mv.col] = nw;
    h ^= ZOBRIST.cell[piece_idx(nw)][mv.row][mv.col];

    s.move_count++;

    if (s.check_win(mv.row, mv.col, p)) {
        s.game_over = true;
        s.winner    = p;
    } else if (s.board_full()) {
        s.game_over = true;
        s.winner    = 0;
    }

    // Toggle player
    s.current_player = -p;
    h ^= ZOBRIST.player;

    s.award_strong_if_due(h);
    return s;
}

// ── GameState::check_win ──────────────────────────────────────────────────────
bool GameState::check_win(int row, int col, int player) const {
    static constexpr int dirs[4][2] = {{0,1},{1,0},{1,1},{1,-1}};
    for (auto& d : dirs) {
        int dr = d[0], dc = d[1], cnt = 1;
        for (int sign : {1, -1}) {
            int r = row + sign * dr, c = col + sign * dc;
            while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE
                   && board[r][c] * player > 0) {
                ++cnt; r += sign * dr; c += sign * dc;
            }
        }
        if (cnt >= WIN_LENGTH) return true;
    }
    return false;
}

// ── GameState::board_full ─────────────────────────────────────────────────────
bool GameState::board_full() const {
    for (auto& row : board)
        for (int8_t v : row)
            if (v == EMPTY) return false;
    return true;
}

// ── GameState::award_strong_if_due ───────────────────────────────────────────
void GameState::award_strong_if_due(uint64_t& h) {
    int next = move_count + 1;
    if (next < 6 || (next - 6) % 7 != 0) return;
    for (int pi = 0; pi < 2; ++pi) {
        if (strong_pieces[pi] < 1) {
            h ^= ZOBRIST.strong[pi][strong_pieces[pi]];
            strong_pieces[pi] = 1;
            h ^= ZOBRIST.strong[pi][1];
        }
    }
}

// ── GameState::observation ───────────────────────────────────────────────────
GameState::Obs GameState::observation() const {
    Obs obs{};
    int p    = current_player;
    int pidx = player_idx(p);
    int oidx = 1 - pidx;
    for (int r = 0; r < BOARD_SIZE; ++r)
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int8_t v = board[r][c];
            obs[0][r][c] = (v ==  p)   ? 1.f : 0.f;
            obs[1][r][c] = (v == 2*p)  ? 1.f : 0.f;
            obs[2][r][c] = (v == -p)   ? 1.f : 0.f;
            obs[3][r][c] = (v == -2*p) ? 1.f : 0.f;
            obs[4][r][c] = static_cast<float>(strong_pieces[pidx]);
            obs[5][r][c] = static_cast<float>(strong_pieces[oidx]);
        }
    return obs;
}

// ── GameState::render ─────────────────────────────────────────────────────────
void GameState::render() const {
    auto sym = [](int8_t v) -> char {
        switch (v) {
            case WHITE_STRONG: return 'O';
            case WHITE_REG:    return 'o';
            case EMPTY:        return '.';
            case BLACK_REG:    return 'x';
            case BLACK_STRONG: return 'X';
            default:           return '?';
        }
    };
    std::printf("    ");
    for (int c = 0; c < BOARD_SIZE; ++c) std::printf("%2d ", c);
    std::printf("\n");
    for (int r = 0; r < BOARD_SIZE; ++r) {
        std::printf("%2d  ", r);
        for (int c = 0; c < BOARD_SIZE; ++c) std::printf(" %c ", sym(board[r][c]));
        std::printf("\n");
    }
    const char* pl = current_player == 1 ? "Black" : "White";
    std::printf("Move #%d  Player: %s  Strong[B=%d W=%d]  Hash=%016llx\n",
                move_count + 1, pl, strong_pieces[0], strong_pieces[1],
                (unsigned long long)zobrist_hash);
    if (game_over) {
        if      (winner ==  1) std::printf("Black wins!\n");
        else if (winner == -1) std::printf("White wins!\n");
        else                   std::printf("Draw!\n");
    }
}
