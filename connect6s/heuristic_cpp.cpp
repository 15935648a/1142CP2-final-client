/**
 * Connect6s heuristic bot v3 — alpha-beta with:
 *   - Zobrist hashing + Transposition Table
 *   - Iterative Deepening with time control + Aspiration Windows
 *   - History Heuristic (beta-cutoff move ordering bonus)
 *   - Live/Blind window openness multiplier
 *   - Fork detection (2+ simultaneous level-4 threats)
 *   - Strong piece window bonus (+50%)
 *   - Defense weight 1.5x in eval
 *   - Tuned score table: S_3=1000, S_2=20
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#include <random>
#include <vector>

namespace py = pybind11;

static constexpr int N  = 15;
static constexpr int NN = N * N;

static constexpr int DR4[4] = {0, 1,  1,  1};
static constexpr int DC4[4] = {1, 0,  1, -1};

// ── Score table ───────────────────────────────────────────────────────────────
static constexpr int S_WIN = 100'000'000;
static constexpr int S_5   =   1'000'000;
static constexpr int S_4   =      10'000;
static constexpr int S_3   =       1'000;
static constexpr int S_2   =          20;
static constexpr int S_1   =           1;
static constexpr int WIN_SCORE[7] = {0, S_1, S_2, S_3, S_4, S_5, S_WIN};

// ── Zobrist ───────────────────────────────────────────────────────────────────
static uint64_t ztable[N][N][4]; // piece: 0=B_reg 1=B_str 2=W_reg 3=W_str
static uint64_t zturn;
static uint64_t zstrong[2];
static bool     zobrist_ready = false;

static void init_zobrist() {
    if (zobrist_ready) return;
    std::mt19937_64 rng(0xCAFEBABEDEADBEEFULL);
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++)
            for (int t = 0; t < 4; t++)
                ztable[r][c][t] = rng();
    zturn      = rng();
    zstrong[0] = rng();
    zstrong[1] = rng();
    zobrist_ready = true;
}

static inline int piece_type(int8_t v) {
    if (v ==  1) return 0;
    if (v ==  2) return 1;
    if (v == -1) return 2;
    if (v == -2) return 3;
    return -1;
}

// ── Transposition Table ───────────────────────────────────────────────────────
static constexpr int    TT_SIZE = 1 << 22;
static constexpr int    TT_MASK = TT_SIZE - 1;
static constexpr uint8_t TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2;

struct TTEntry {
    uint64_t hash;
    int      score;
    int16_t  action;
    int8_t   depth;
    uint8_t  flag;
};
static TTEntry tt[TT_SIZE];

// ── History Heuristic ─────────────────────────────────────────────────────────
static int history_table[2][NN];  // [player_idx][r*N+c]

// ── Killer Move table ─────────────────────────────────────────────────────────
static constexpr int MAX_PLY = 20;
struct KillerEntry { int8_t r, c; bool is_str, valid; };
static KillerEntry killers[MAX_PLY][2];

static void clear_search_tables() {
    std::memset(tt,            0, sizeof(tt));
    std::memset(history_table, 0, sizeof(history_table));
    std::memset(killers,       0, sizeof(killers));
}

// ── Board ─────────────────────────────────────────────────────────────────────
struct Board {
    int8_t   cell[N][N];
    int      player;
    int      move_count;
    int      strong[2];
    uint64_t zhash;

    Board() : player(1), move_count(0), zhash(0) {
        std::memset(cell, 0, sizeof(cell));
        strong[0] = strong[1] = 0;
    }

    int  pidx()       const { return player > 0 ? 0 : 1; }
    bool has_strong() const { return strong[pidx()] > 0; }

    static bool owns(int8_t v, int p) { return p > 0 ? v > 0 : v < 0; }

    bool can_strong_at(int r, int c) const {
        if (!has_strong()) return false;
        int8_t v = cell[r][c];
        return v == 0 || v == 1 || v == -1;
    }
    bool my_reg (int r, int c) const {
        return player > 0 ? cell[r][c] ==  1 : cell[r][c] == -1;
    }
    bool opp_reg(int r, int c) const {
        return player > 0 ? cell[r][c] == -1 : cell[r][c] ==  1;
    }

    Board apply(int r, int c, bool is_str) const {
        Board nx = *this;
        int old_t = piece_type(cell[r][c]);
        if (old_t >= 0) nx.zhash ^= ztable[r][c][old_t];
        nx.cell[r][c] = is_str ? (int8_t)(player > 0 ? 2 : -2)
                                : (int8_t)player;
        nx.zhash ^= ztable[r][c][piece_type(nx.cell[r][c])];
        if (is_str) {
            nx.zhash ^= zstrong[nx.pidx()];
            nx.strong[nx.pidx()]--;
        }
        nx.zhash ^= zturn;
        nx.player = -player;
        nx.move_count++;
        int nxt = nx.move_count + 1;
        if (nxt >= 6 && (nxt - 6) % 7 == 0) {
            if (nx.strong[0] < 1) { nx.zhash ^= zstrong[0]; nx.strong[0] = 1; }
            if (nx.strong[1] < 1) { nx.zhash ^= zstrong[1]; nx.strong[1] = 1; }
        }
        return nx;
    }

    int winner() const {
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                int8_t v = cell[r][c];
                if (!v) continue;
                int p = v > 0 ? 1 : -1;
                for (int d = 0; d < 4; d++) {
                    int pr = r - DR4[d], pc = c - DC4[d];
                    if (pr >= 0 && pr < N && pc >= 0 && pc < N
                            && owns(cell[pr][pc], p)) continue;
                    int len = 1;
                    for (int k = 1; k < 6; k++) {
                        int nr = r + DR4[d]*k, nc = c + DC4[d]*k;
                        if (nr < 0 || nr >= N || nc < 0 || nc >= N
                                || !owns(cell[nr][nc], p)) break;
                        len++;
                    }
                    if (len >= 6) return p;
                }
            }
        }
        return 0;
    }
};

static uint64_t compute_initial_hash(const Board& b) {
    uint64_t h = (b.player == 1) ? zturn : 0ULL;
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++) {
            int t = piece_type(b.cell[r][c]);
            if (t >= 0) h ^= ztable[r][c][t];
        }
    if (b.strong[0] > 0) h ^= zstrong[0];
    if (b.strong[1] > 0) h ^= zstrong[1];
    return h;
}

// ── Evaluation ────────────────────────────────────────────────────────────────

// Window openness: checks if cells just beyond both ends of the 6-window are
// accessible (empty). Open windows are more likely to be completable.
static int eval_side(const Board& b, int p) {
    int score = 0;
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            for (int d = 0; d < 4; d++) {
                // Window end
                int er = r + DR4[d]*5, ec = c + DC4[d]*5;
                if (er < 0 || er >= N || ec < 0 || ec >= N) continue;

                int  n_own = 0, n_opp = 0;
                bool has_str = false;
                for (int k = 0; k < 6; k++) {
                    int8_t v = b.cell[r + DR4[d]*k][c + DC4[d]*k];
                    if (!v) continue;
                    if (Board::owns(v, p)) {
                        n_own++;
                        if (std::abs((int)v) == 2) has_str = true;
                    } else n_opp++;
                }
                if (n_opp > 0) continue;
                if (n_own == 6) return S_WIN;

                int contrib = WIN_SCORE[n_own];

                // Strong piece bonus: strong pieces can't be captured
                if (has_str && n_own > 0) contrib = contrib * 3 / 2;

                // Live/Blind openness multiplier:
                // Check cell before window start and after window end.
                // If both ends are open (empty), the window can grow in both
                // directions → higher completion potential.
                int open_ends = 0;
                {
                    int br = r - DR4[d], bc = c - DC4[d];
                    if (br >= 0 && br < N && bc >= 0 && bc < N
                            && b.cell[br][bc] == 0) open_ends++;
                    int ar = er + DR4[d], ac = ec + DC4[d];
                    if (ar >= 0 && ar < N && ac >= 0 && ac < N
                            && b.cell[ar][ac] == 0) open_ends++;
                }
                if (open_ends == 2 && n_own == 2) contrib = S_2 * 15;  // live-2 early warning: ~300
                else if (open_ends == 2) contrib = contrib * 3 / 2;   // live 3+: +50%
                else if (open_ends == 0 && n_own >= 4) contrib /= 2;  // dead high-threat: -50%

                score += contrib;
            }
        }
    }
    return score;
}

// Defense weighted 1.2× (reduced from 1.5× to encourage proactive play)
static int heuristic(const Board& b) {
    return eval_side(b, b.player) - 6 * eval_side(b, -b.player) / 5;
}

static int cell_value(const Board& b, int r, int c, int p) {
    int score = 0;
    for (int d = 0; d < 4; d++) {
        for (int offset = 0; offset < 6; offset++) {
            int sr = r - DR4[d]*offset, sc = c - DC4[d]*offset;
            int er = sr + DR4[d]*5,    ec = sc + DC4[d]*5;
            if (sr < 0 || sr >= N || sc < 0 || sc >= N) continue;
            if (er < 0 || er >= N || ec < 0 || ec >= N) continue;
            int n_own = 0, n_opp = 0;
            for (int k = 0; k < 6; k++) {
                int kr = sr + DR4[d]*k, kc = sc + DC4[d]*k;
                if (kr == r && kc == c) continue;
                int8_t v = b.cell[kr][kc];
                if (!v) continue;
                if (Board::owns(v, p)) n_own++; else n_opp++;
            }
            if (n_opp > 0) continue;
            score += WIN_SCORE[std::min(n_own + 1, 6)];
        }
    }
    return score;
}

// Count distinct DIRECTIONS that have a threat of >= min_level after placing at (r,c).
// One direction = one threat regardless of how many overlapping windows cover it.
// True fork requires threats in 2+ different directions (can't be blocked by one move).
static int count_fork_directions(const Board& b, int r, int c, int p, int min_level) {
    int dirs = 0;
    for (int d = 0; d < 4; d++) {
        bool found = false;
        for (int offset = 0; offset < 6 && !found; offset++) {
            int sr = r - DR4[d]*offset, sc = c - DC4[d]*offset;
            int er = sr + DR4[d]*5,    ec = sc + DC4[d]*5;
            if (sr < 0 || sr >= N || sc < 0 || sc >= N) continue;
            if (er < 0 || er >= N || ec < 0 || ec >= N) continue;
            int n_own = 0, n_opp = 0;
            for (int k = 0; k < 6; k++) {
                int kr = sr + DR4[d]*k, kc = sc + DC4[d]*k;
                if (kr == r && kc == c) continue;
                int8_t v = b.cell[kr][kc];
                if (!v) continue;
                if (Board::owns(v, p)) n_own++; else n_opp++;
            }
            if (n_opp > 0) continue;
            if (n_own + 1 >= min_level) found = true;
        }
        if (found) dirs++;
    }
    return dirs;
}

// ── Move generation ───────────────────────────────────────────────────────────
struct Move { int r, c; bool is_str; };

static std::vector<std::pair<int,Move>> gen_moves(const Board& b) {
    bool near[N][N] = {};
    bool any = false;
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            if (!b.cell[r][c]) continue;
            any = true;
            // Use radius-3 for opponent pieces so we generate candidates
            // near distant threats before they become critical.
            // Use radius-2 for own pieces (tighter, saves candidate budget).
            int rad = Board::owns(b.cell[r][c], b.player) ? 2 : 3;
            for (int dr = -rad; dr <= rad; dr++)
            for (int dc = -rad; dc <= rad; dc++) {
                int nr = r+dr, nc = c+dc;
                if (nr >= 0 && nr < N && nc >= 0 && nc < N)
                    near[nr][nc] = true;
            }
        }
    }
    if (!any) return {{0, {N/2, N/2, false}}};

    static constexpr int FORK_BONUS   = S_5;       // double-4 = near-forced-win → S_5
    static constexpr int FORK3_BONUS  = 3 * S_3;   // double-live-3 → dangerous setup
    int pidx = b.player > 0 ? 0 : 1;

    std::vector<std::pair<int,Move>> moves;
    moves.reserve(150);

    auto make_score = [](int atk, int def, int bonus = 0) -> int {
        if (atk >= S_WIN) return 3*S_WIN + atk;
        if (def >= S_WIN) return 2*S_WIN + def;
        return 2*atk + def + bonus;  // 2x atk: prefer extensions over blocks
    };

    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            int8_t v = b.cell[r][c];
            // History heuristic bonus (right-shift to keep in range)
            int hist = history_table[pidx][r*N + c] >> 4;

            if (v == 0 && near[r][c]) {
                int atk = cell_value(b, r, c,  b.player);
                int def = cell_value(b, r, c, -b.player);
                int bonus = hist;
                if (atk < S_WIN && def < S_WIN) {
                    if (count_fork_directions(b, r, c, b.player, 4) >= 2)
                        bonus += FORK_BONUS;
                    else if (count_fork_directions(b, r, c, b.player, 3) >= 2)
                        bonus += FORK3_BONUS;
                }
                moves.push_back({make_score(atk, def, bonus), {r, c, false}});
            }

            if (b.can_strong_at(r, c)) {
                if (v == 0 && near[r][c]) {
                    int atk = cell_value(b, r, c,  b.player);
                    int def = cell_value(b, r, c, -b.player);
                    int bonus = -S_3/2 + hist;  // strong penalty: save strong pieces for captures
                    if (atk < S_WIN && def < S_WIN) {
                        if (count_fork_directions(b, r, c, b.player, 4) >= 2)
                            bonus += FORK_BONUS;
                        else if (count_fork_directions(b, r, c, b.player, 3) >= 2)
                            bonus += FORK3_BONUS;
                    }
                    moves.push_back({make_score(atk, def, bonus), {r, c, true}});
                } else if (b.opp_reg(r, c)) {
                    // Strong capture: our uncapturable piece replaces opponent's piece.
                    // Boost atk ×3/2 (strong piece can't be recaptured).
                    int atk = cell_value(b, r, c,  b.player) * 3 / 2;
                    int def = cell_value(b, r, c, -b.player);
                    int bonus = hist;
                    // Fork bonus: capture creates multi-direction attack
                    if (atk < S_WIN && def < S_WIN) {
                        if (count_fork_directions(b, r, c, b.player, 4) >= 2)
                            bonus += FORK_BONUS;
                        else if (count_fork_directions(b, r, c, b.player, 3) >= 2)
                            bonus += FORK3_BONUS;
                    }
                    // Chain-break bonus: disrupting opponent's 4+ threat
                    if (def >= S_4) bonus += S_4;
                    moves.push_back({make_score(atk, 3*def/2, bonus), {r, c, true}});
                } else if (b.my_reg(r, c) && near[r][c]) {
                    // Bug fix: include atk (upgrade makes own piece uncapturable)
                    int atk = cell_value(b, r, c,  b.player);
                    int def = cell_value(b, r, c, -b.player);
                    moves.push_back({make_score(atk/2, def), {r, c, true}});
                }
            }
        }
    }

    std::sort(moves.begin(), moves.end(),
              [](const auto& a, const auto& b_) { return a.first > b_.first; });

    if (moves.size() <= 30) return moves;

    // Keep top-30, but rescue any move outside top-30 that blocks a
    // critical threat (def >= S_4 — opponent has 4+ in a window there).
    std::vector<std::pair<int,Move>> result;
    result.reserve(35);
    for (int i = 0; i < 30; i++) result.push_back(moves[i]);
    for (int i = 30; i < (int)moves.size(); i++) {
        auto& [sc, m] = moves[i];
        if (cell_value(b, m.r, m.c, -b.player) >= S_4)
            result.push_back(moves[i]);
    }
    return result;
}

// ── Alpha-beta negamax with TT + History ─────────────────────────────────────
static bool time_up_flag = false;
static std::chrono::steady_clock::time_point search_deadline;
static int eval_counter = 0;

static inline void push_tt_best_front(std::vector<std::pair<int,Move>>& moves,
                                       int16_t tt_action) {
    if (tt_action < 0) return;
    for (auto it = moves.begin(); it != moves.end(); ++it) {
        int enc = it->second.is_str ? (NN + it->second.r*N + it->second.c)
                                    : (it->second.r*N + it->second.c);
        if (enc == (int)tt_action) {
            auto entry = *it;
            moves.erase(it);
            moves.insert(moves.begin(), entry);
            return;
        }
    }
}

static inline void push_killers_front(std::vector<std::pair<int,Move>>& moves, int ply) {
    if (ply >= MAX_PLY || (int)moves.size() < 2) return;
    int insert_pos = 1;  // slot 0 reserved for TT best
    for (int ki = 0; ki < 2; ki++) {
        auto& k = killers[ply][ki];
        if (!k.valid) continue;
        for (int i = insert_pos; i < (int)moves.size(); i++) {
            auto& m = moves[i].second;
            if (m.r == k.r && m.c == k.c && m.is_str == k.is_str) {
                if (i != insert_pos) {
                    auto entry = moves[i];
                    moves.erase(moves.begin() + i);
                    moves.insert(moves.begin() + insert_pos, entry);
                }
                insert_pos++;
                break;
            }
        }
    }
}

static int negamax(const Board& b, int depth, int ply, int alpha, int beta) {
    if (time_up_flag) return 0;

    uint64_t h = b.zhash;
    int orig_alpha = alpha;

    // TT lookup
    TTEntry& te = tt[h & TT_MASK];
    int16_t  tt_action = -1;
    if (te.hash == h) {
        tt_action = te.action;
        if (te.depth >= depth) {
            if      (te.flag == TT_EXACT) return te.score;
            else if (te.flag == TT_LOWER) alpha = std::max(alpha, te.score);
            else if (te.flag == TT_UPPER) beta  = std::min(beta,  te.score);
            if (alpha >= beta) return te.score;
        }
    }

    int w = b.winner();
    if (w != 0) return -(S_WIN + depth);

    if (depth == 0) {
        if ((++eval_counter & 0xFF) == 0) {
            if (std::chrono::steady_clock::now() >= search_deadline)
                time_up_flag = true;
        }
        return time_up_flag ? 0 : heuristic(b);
    }

    auto moves = gen_moves(b);
    if (moves.empty()) return 0;

    push_tt_best_front(moves, tt_action);
    push_killers_front(moves, ply);

    int     best = INT_MIN / 2;
    int16_t best_action = -1;
    int     pidx = (b.player == 1) ? 0 : 1;

    for (auto& [sc, m] : moves) {
        if (time_up_flag) break;
        Board nx  = b.apply(m.r, m.c, m.is_str);
        int   val = -negamax(nx, depth - 1, ply + 1, -beta, -alpha);
        if (time_up_flag) break;
        if (val > best) {
            best = val;
            best_action = (int16_t)(m.is_str ? (NN + m.r*N + m.c)
                                              : (m.r*N + m.c));
        }
        if (best > alpha) alpha = best;
        if (alpha >= beta) {
            // History + killer update on cutoff (non-captures only)
            if (!m.is_str) {
                history_table[pidx][m.r*N + m.c] += depth * depth;
                if (ply < MAX_PLY) {
                    killers[ply][1] = killers[ply][0];
                    killers[ply][0] = {(int8_t)m.r, (int8_t)m.c, m.is_str, true};
                }
            }
            break;
        }
    }

    if (!time_up_flag && best > INT_MIN / 2) {
        te.hash   = h;
        te.score  = best;
        te.depth  = (int8_t)std::min(depth, 127);
        te.action = best_action;
        if      (best <= orig_alpha) te.flag = TT_UPPER;
        else if (best >= beta)       te.flag = TT_LOWER;
        else                         te.flag = TT_EXACT;
    }
    return best;
}

// ── Python-facing class ───────────────────────────────────────────────────────
class HeuristicBot {
public:
    int best_action(
        py::array_t<int8_t> board_arr,
        int player, int move_count,
        int strong_b, int strong_w,
        int depth = 4
    ) {
        Board b = make_board(board_arr, player, move_count, strong_b, strong_w);
        search_deadline = std::chrono::steady_clock::now() + std::chrono::hours(24);
        time_up_flag = false;
        eval_counter = 0;
        return search_at_depth(b, depth, -S_WIN*2, S_WIN*2, nullptr);
    }

    int best_action_timed(
        py::array_t<int8_t> board_arr,
        int player, int move_count,
        int strong_b, int strong_w,
        double seconds = 5.0
    ) {
        Board b = make_board(board_arr, player, move_count, strong_b, strong_w);
        search_deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds((long long)(seconds * 920));
        time_up_flag = false;
        eval_counter = 0;
        clear_search_tables();

        // Fallback: first candidate from gen_moves
        int best_idx = 0;
        {
            auto cands = gen_moves(b);
            if (!cands.empty()) {
                auto& m0 = cands[0].second;
                best_idx = m0.is_str ? (NN + m0.r*N + m0.c) : (m0.r*N + m0.c);
            }
        }

        // Iterative deepening with aspiration windows
        // Delta starts at ±50,000 (5×S_4); grows ×4 on each fail until full window
        int prev_score = 0;

        for (int d = 1; d <= 20 && !time_up_flag; d++) {
            int score = 0;
            int idx;

            if (d <= 3) {
                idx = search_at_depth(b, d, -S_WIN*2, S_WIN*2, &score);
            } else {
                int delta = S_4 * 5;  // ±50,000
                int alpha = prev_score - delta;
                int beta  = prev_score + delta;
                idx = search_at_depth(b, d, alpha, beta, &score);

                // Grow window ×4 on fail until it covers
                while (!time_up_flag && (score <= alpha || score >= beta)) {
                    delta *= 4;
                    alpha = (score <= alpha) ? prev_score - delta : alpha;
                    beta  = (score >= beta)  ? prev_score + delta : beta;
                    idx = search_at_depth(b, d, alpha, beta, &score);
                }
            }

            if (!time_up_flag) {
                best_idx   = idx;
                prev_score = score;
            }
        }
        return best_idx;
    }

private:
    Board make_board(py::array_t<int8_t> board_arr,
                     int player, int move_count,
                     int strong_b, int strong_w) {
        Board b;
        auto buf = board_arr.unchecked<1>();
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++)
                b.cell[r][c] = buf[r*N + c];
        b.player     = player;
        b.move_count = move_count;
        b.strong[0]  = strong_b;
        b.strong[1]  = strong_w;
        b.zhash      = compute_initial_hash(b);
        return b;
    }

    // Root search at given depth with explicit alpha/beta window.
    // Returns best action index; writes root score to *out_score if not null.
    int search_at_depth(const Board& b, int depth,
                        int alpha, int beta, int* out_score) {
        auto candidates = gen_moves(b);
        if (candidates.empty()) { if (out_score) *out_score = 0; return 0; }

        // Immediate win/block
        for (auto& [sc, m] : candidates) {
            if (sc >= 3*S_WIN) {
                int idx = m.is_str ? (NN + m.r*N + m.c) : (m.r*N + m.c);
                if (out_score) *out_score = S_WIN;
                return idx;
            }
        }

        // TT best move first
        {
            TTEntry& te = tt[b.zhash & TT_MASK];
            if (te.hash == b.zhash) push_tt_best_front(candidates, te.action);
        }

        int  best_val = INT_MIN / 2;
        Move best_m   = candidates[0].second;

        for (auto& [sc, m] : candidates) {
            if (time_up_flag) break;
            Board nx  = b.apply(m.r, m.c, m.is_str);
            int   val = -negamax(nx, depth - 1, 1, -beta, -alpha);
            if (time_up_flag) break;
            if (val > best_val) {
                best_val = val;
                best_m   = m;
            }
            if (best_val > alpha) alpha = best_val;
            if (alpha >= beta) break;
        }

        if (out_score) *out_score = best_val;
        return best_m.is_str ? (NN + best_m.r*N + best_m.c)
                             : (best_m.r*N + best_m.c);
    }
};

PYBIND11_MODULE(connect6s_heuristic_cpp, m) {
    init_zobrist();
    m.doc() = "Connect6s heuristic bot v3 — TT + ID + aspiration + history + live/blind";
    py::class_<HeuristicBot>(m, "HeuristicBot")
        .def(py::init<>())
        .def("best_action", &HeuristicBot::best_action,
             py::arg("board"), py::arg("player"), py::arg("move_count"),
             py::arg("strong_b"), py::arg("strong_w"), py::arg("depth") = 4)
        .def("best_action_timed", &HeuristicBot::best_action_timed,
             py::arg("board"), py::arg("player"), py::arg("move_count"),
             py::arg("strong_b"), py::arg("strong_w"), py::arg("seconds") = 5.0);
}
