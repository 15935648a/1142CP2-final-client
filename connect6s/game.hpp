#pragma once
#include <array>
#include <cstdint>
#include <vector>

// ── Constants ────────────────────────────────────────────────────────────────
static constexpr int BOARD_SIZE  = 15;
static constexpr int WIN_LENGTH  = 6;
static constexpr int ACTION_SIZE = BOARD_SIZE * BOARD_SIZE * 2;  // regular + strong

// Cell values stored in board
enum Cell : int8_t {
    WHITE_STRONG = -2,
    WHITE_REG    = -1,
    EMPTY        =  0,
    BLACK_REG    =  1,
    BLACK_STRONG =  2,
};

// ── Zobrist tables ────────────────────────────────────────────────────────────
// piece_idx: WHITE_STRONG(-2)→0, WHITE_REG(-1)→1, BLACK_REG(1)→2, BLACK_STRONG(2)→3
// player_idx: black=0, white=1   |   strong count 0/1 per player
struct ZobristTables {
    uint64_t cell  [4][BOARD_SIZE][BOARD_SIZE];
    uint64_t player;                  // XOR when it is White's turn
    uint64_t strong[2][2];            // [player_idx][count]
    uint64_t init_hash;               // hash of empty board, Black to move, both strong=0
};
extern const ZobristTables ZOBRIST;

// piece_idx helper: maps {-2,-1,1,2} → {0,1,2,3}
inline int piece_idx(int8_t v) { return v < 0 ? (v + 2) : (v + 1); }

// ── Move ─────────────────────────────────────────────────────────────────────
struct Move {
    int  row;
    int  col;
    bool is_strong;

    int to_index() const {
        int flat = row * BOARD_SIZE + col;
        return is_strong ? flat + BOARD_SIZE * BOARD_SIZE : flat;
    }

    static Move from_index(int idx) {
        if (idx >= BOARD_SIZE * BOARD_SIZE)
            return {(idx - BOARD_SIZE * BOARD_SIZE) / BOARD_SIZE,
                    (idx - BOARD_SIZE * BOARD_SIZE) % BOARD_SIZE, true};
        return {idx / BOARD_SIZE, idx % BOARD_SIZE, false};
    }

    bool operator==(const Move& o) const {
        return row == o.row && col == o.col && is_strong == o.is_strong;
    }
};

// ── GameState ─────────────────────────────────────────────────────────────────
class GameState {
public:
    using Board = std::array<std::array<int8_t, BOARD_SIZE>, BOARD_SIZE>;

    Board    board{};
    int      current_player{1};        // 1=Black, -1=White
    int      move_count{0};
    int      strong_pieces[2]{0, 0};   // [black_idx=0, white_idx=1]
    bool     game_over{false};
    int      winner{0};                // 1, -1, or 0 (draw / ongoing)
    uint64_t zobrist_hash{0};

    GameState() { clear(); }

    void clear();

    // ── Accessors ─────────────────────────────────────────────────────────────
    int8_t cell_val(int r, int c) const { return board[r][c]; }

    static int player_idx(int player) { return player == 1 ? 0 : 1; }
    int strong_for(int player) const  { return strong_pieces[player_idx(player)]; }

    // ── Move generation ───────────────────────────────────────────────────────
    std::vector<Move> legal_moves() const;

    // ── State transition (returns new state; does not mutate) ─────────────────
    GameState apply(Move mv) const;

    // ── Observation for neural network: float[6][BOARD_SIZE][BOARD_SIZE] ──────
    using Obs = std::array<std::array<std::array<float, BOARD_SIZE>, BOARD_SIZE>, 6>;
    Obs observation() const;

    void render() const;

private:
    bool check_win(int row, int col, int player) const;
    bool board_full() const;
    void award_strong_if_due(uint64_t& h);
};
