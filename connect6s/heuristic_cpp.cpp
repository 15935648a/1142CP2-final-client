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
struct Board;
static void update_scores_at(const Board& before, Board& after, int r, int c);  // fwd

struct Board {
    int8_t   cell[N][N];
    int      player;
    int      move_count;
    int      strong[2];
    uint64_t zhash;
    int      score[2];   // score[0]=eval for player1(black), score[1]=eval for player2(white)
    int8_t   last_r, last_c;  // last placed cell; -1 on empty board

    Board() : player(1), move_count(0), zhash(0), last_r(-1), last_c(-1) {
        std::memset(cell, 0, sizeof(cell));
        strong[0] = strong[1] = 0;
        score[0]  = score[1]  = 0;
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
        nx.last_r = (int8_t)r;
        nx.last_c = (int8_t)c;
        update_scores_at(*this, nx, r, c);
        return nx;
    }

    int winner() const {
        if (last_r < 0) return 0;  // empty board
        int8_t v = cell[last_r][last_c];
        if (!v) return 0;
        int p = v > 0 ? 1 : -1;
        for (int d = 0; d < 4; d++) {
            int cnt = 1;
            for (int sign = 1; sign >= -1; sign -= 2) {
                int rr = last_r + sign*DR4[d], cc = last_c + sign*DC4[d];
                while (rr >= 0 && rr < N && cc >= 0 && cc < N && owns(cell[rr][cc], p)) {
                    cnt++;
                    rr += sign*DR4[d];
                    cc += sign*DC4[d];
                }
            }
            if (cnt >= 6) return p;
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

// Score contribution of a single 6-cell window starting at (sr,sc) direction d for player p.
static int eval_one_window(const Board& b, int sr, int sc, int d, int p) {
    int er = sr + DR4[d]*5, ec = sc + DC4[d]*5;
    if (sr < 0 || sr >= N || sc < 0 || sc >= N) return 0;
    if (er < 0 || er >= N || ec < 0 || ec >= N) return 0;

    int  n_own = 0, n_opp = 0;
    bool has_str = false, opp_has_reg = false;
    for (int k = 0; k < 6; k++) {
        int8_t v = b.cell[sr + DR4[d]*k][sc + DC4[d]*k];
        if (!v) continue;
        if (Board::owns(v, p)) { n_own++; if (std::abs((int)v) == 2) has_str = true; }
        else { n_opp++; if (std::abs((int)v) == 1) opp_has_reg = true; }
    }
    if (n_opp > 0) {
        // p can win via strong capture: 5 own + 1 capturable opp regular + p has strong piece
        if (n_own == 5 && n_opp == 1 && opp_has_reg && b.strong[(p > 0) ? 0 : 1] > 0)
            return S_5;
        return 0;
    }
    if (n_own == 6) return S_WIN;

    int contrib = WIN_SCORE[n_own];
    if (has_str && n_own > 0) contrib = contrib * 3 / 2;

    int open_ends = 0;
    int br = sr - DR4[d], bc = sc - DC4[d];
    if (br >= 0 && br < N && bc >= 0 && bc < N && b.cell[br][bc] == 0) open_ends++;
    int ar = er + DR4[d], ac = ec + DC4[d];
    if (ar >= 0 && ar < N && ac >= 0 && ac < N && b.cell[ar][ac] == 0) open_ends++;

    if      (open_ends == 2 && n_own == 2) return S_2 * 15;
    else if (open_ends == 2 && n_own == 3) return contrib * 4;
    else if (open_ends == 2)               return contrib * 3 / 2;
    else if (open_ends == 0 && n_own >= 3) return contrib / 2;
    return contrib;
}

// Update b.score[] incrementally after placing at (r,c).
// `before` = board state before cell change, `after` = board with new cell already set.
// Patches all windows that contain (r,c) or use it as an openness-check cell.
static void update_scores_at(const Board& before, Board& after, int r, int c) {
    for (int pi = 0; pi < 2; pi++) {
        int p = (pi == 0) ? 1 : -1;
        for (int d = 0; d < 4; d++) {
            // k = 0..5: windows containing (r,c); k=-1,k=6: windows where (r,c) is adjacency cell
            for (int k = -1; k <= 6; k++) {
                int sr = r - DR4[d]*k, sc = c - DC4[d]*k;
                after.score[pi] -= eval_one_window(before, sr, sc, d, p);
                after.score[pi] += eval_one_window(after,  sr, sc, d, p);
            }
        }
    }
}

static int eval_side(const Board& b, int p);  // forward declaration

static void compute_initial_scores(Board& b) {
    b.score[0] = eval_side(b,  1);
    b.score[1] = eval_side(b, -1);
}

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
                bool has_str = false, opp_has_reg = false;
                for (int k = 0; k < 6; k++) {
                    int8_t v = b.cell[r + DR4[d]*k][c + DC4[d]*k];
                    if (!v) continue;
                    if (Board::owns(v, p)) {
                        n_own++;
                        if (std::abs((int)v) == 2) has_str = true;
                    } else { n_opp++; if (std::abs((int)v) == 1) opp_has_reg = true; }
                }
                if (n_opp > 0) {
                    if (n_own == 5 && n_opp == 1 && opp_has_reg && b.strong[(p > 0) ? 0 : 1] > 0)
                        { score += S_5; }
                    continue;
                }
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
                if      (open_ends == 2 && n_own == 2) contrib = S_2 * 15; // live-2: ~300
                else if (open_ends == 2 && n_own == 3) contrib *= 4;  // live-3: both ends open = dangerous
                else if (open_ends == 2)              contrib = contrib * 3 / 2; // live 4+: +50%
                // open_ends == 1: keep base contrib (half-open, no multiplier)
                else if (open_ends == 0 && n_own >= 3) contrib /= 2;  // dead 3+: -50%

                score += contrib;
            }
        }
    }
    return score;
}

static int heuristic(const Board& b) {
    int pi = b.player > 0 ? 0 : 1;
    return b.score[pi] - 6 * b.score[1 - pi] / 5;
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
            int contrib = WIN_SCORE[std::min(n_own + 1, 6)];
            // Openness multiplier: live threats (both ends open) are far more
            // dangerous than half-open or dead ones. Without this, cell_value
            // under-rates blocking moves vs extending own pieces (which
            // naturally count windows in all 4 directions).
            if (n_own + 1 >= 3) {
                int br = sr - DR4[d], bc = sc - DC4[d];
                int ar = er + DR4[d], ac = ec + DC4[d];
                bool open_lo = (br>=0&&br<N&&bc>=0&&bc<N && b.cell[br][bc]==0);
                bool open_hi = (ar>=0&&ar<N&&ac>=0&&ac<N && b.cell[ar][ac]==0);
                if      (open_lo && open_hi) contrib *= 2;  // live: double threat value
                else if (!open_lo && !open_hi) contrib /= 2; // dead: halve
            }
            score += contrib;
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
    static constexpr int FORK3_BONUS  = 8 * S_3;    // double-live-3 → dangerous setup
    int pidx = b.player > 0 ? 0 : 1;
	int next_refresh = 6;
	while (next_refresh <= b.move_count + 1) {
		next_refresh += 7;
	}
	bool is_last_chance = (next_refresh <= b.move_count + 3);
	int strong_penalty = is_last_chance ? 0 : -S_3/2;

    std::vector<std::pair<int,Move>> moves;
    moves.reserve(150);

    auto make_score = [](int atk, int def, int bonus = 0) -> int {
        if (atk >= S_WIN) return 3*S_WIN + atk;
        if (def >= S_WIN) return 2*S_WIN + def;
        return atk + 6*def/5 + bonus;  // 1.2x def
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
                // Connectivity bonus: adjacent to own piece → builds connected structure
                for (int dr = -1; dr <= 1 && !(bonus & (S_3/2)); dr++)
                    for (int dc = -1; dc <= 1; dc++) {
                        if (!dr && !dc) continue;
                        int nr=r+dr, nc=c+dc;
                        if (nr>=0&&nr<N&&nc>=0&&nc<N&&Board::owns(b.cell[nr][nc],b.player))
                            { bonus += S_3/2; break; }
                    }
                // Synergy bonus: dual-purpose move (both attacks and defends)
                if (atk >= S_3 && def >= S_3) bonus += S_3;
                if (atk < S_WIN && def < S_WIN) {
                    // Own fork bonus: we create double-attack
                    if (count_fork_directions(b, r, c, b.player, 4) >= 2)
                        bonus += FORK_BONUS;
                    else if (count_fork_directions(b, r, c, b.player, 3) >= 2)
                        bonus += FORK3_BONUS;
                    // Opponent fork block: prevent opponent double-threat
                    if (count_fork_directions(b, r, c, -b.player, 4) >= 2)
                        bonus += FORK_BONUS;
                    else if (count_fork_directions(b, r, c, -b.player, 3) >= 2)
                        bonus += FORK3_BONUS;
                }
                moves.push_back({make_score(atk, def, bonus), {r, c, false}});
            }

            if (b.can_strong_at(r, c)) {
                if (v == 0 && near[r][c]) {
                    // Use actual eval diff: partial block of 4-in-a-row shouldn't
                    // score S_5 when the line remains threatening from the other end.
                    Board after_blk = b.apply(r, c, true);
                    int opp_pi_b = (b.player > 0) ? 1 : 0;
                    int my_pi_b  = 1 - opp_pi_b;
                    int def = std::max(0, b.score[opp_pi_b] - after_blk.score[opp_pi_b]);
                    int atk = std::max(0, after_blk.score[my_pi_b] - b.score[my_pi_b]);
                    int bonus = strong_penalty + hist;  // strong penalty: waived on last
                    if (atk < S_WIN && def < S_WIN) {
                        if (count_fork_directions(b, r, c, b.player, 4) >= 2)
                            bonus += FORK_BONUS;
                        else if (count_fork_directions(b, r, c, b.player, 3) >= 2)
                            bonus += FORK3_BONUS;
                    }
                    moves.push_back({make_score(atk, def, bonus), {r, c, true}});
                } else if (b.opp_reg(r, c)) {
                    // Strong capture: use actual eval diff for accurate disruption.
                    // cell_value() only measures "new piece gain", not "existing piece removal".
                    // Board::apply() incremental update is O(64 patches) — cheap per candidate.
                    Board after_cap = b.apply(r, c, true);
                    int opp_pi = (b.player > 0) ? 1 : 0;
                    int my_pi  = 1 - opp_pi;
                    int disruption = std::max(0, b.score[opp_pi] - after_cap.score[opp_pi]);
                    int gain       = std::max(0, after_cap.score[my_pi] - b.score[my_pi]);
                    int atk = gain * 3 / 2;
                    int def = disruption;
                    int bonus = hist;
                    if (atk < S_WIN && def < S_WIN) {
                        // Unblockable fork: strong piece fork is lethal (×1.5 bonus)
                        if (count_fork_directions(b, r, c, b.player, 4) >= 2)
                            bonus += FORK_BONUS * 3 / 2;
                        else if (count_fork_directions(b, r, c, b.player, 3) >= 2)
                            bonus += FORK3_BONUS * 3 / 2;
                    }
                    // High-threat line bonus: capture forms 5-in-a-row or 4-in-a-row
                    if      (atk >= S_5) bonus += S_5;
                    else if (atk >= S_4) bonus += S_4 * 5;
                    // Chain-break bonus: graduated by opponent line strength
                    if      (def >= S_4) bonus += S_4 * 3;   // disrupts 4+: 30K
                    else if (def >= S_3) bonus += S_3 * 3;   // disrupts 3+: 3K (new)
                    moves.push_back({make_score(atk, 2*def, bonus), {r, c, true}});
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
        if (cell_value(b, m.r, m.c, -b.player) >= S_4
            || count_fork_directions(b, m.r, m.c, -b.player, 3) >= 2)
            result.push_back(moves[i]);
    }
    return result;
}

// Shared time-budget state (used by VCF, VCT, and alpha-beta)
static bool time_up_flag = false;
static std::chrono::steady_clock::time_point search_deadline;
static int eval_counter = 0;

// ── VCF (Victory by Consecutive Forcing / 追四勝) ────────────────────────────
// Finds forced win via repeated 5-threats. Attacker creates 5-chain each move;
// defender forced to fill the one gap. Terminates when attacker creates dual
// simultaneous 5-threats (opponent can only block one → win through the other).

static constexpr int VCF_MAX_PLY = 14;  // 7 attack+defense pairs

// After virtually placing new_val at (r,c) for player p, collect all empty gap
// cells opponent must fill to prevent immediate 6-in-a-row next turn.
static std::vector<std::pair<int,int>> vcf_gaps(
    const Board& b, int r, int c, int p, int8_t new_val)
{
    std::vector<std::pair<int,int>> gaps;
    for (int d = 0; d < 4; d++) {
        for (int k = 0; k < 6; k++) {
            int sr = r - DR4[d]*k, sc = c - DC4[d]*k;
            int er = sr + DR4[d]*5, ec = sc + DC4[d]*5;
            if (sr<0||sr>=N||sc<0||sc>=N||er<0||er>=N||ec<0||ec>=N) continue;
            int n_own=0, n_opp=0, gr=-1, gc=-1;
            for (int j = 0; j < 6; j++) {
                int kr = sr+DR4[d]*j, kc = sc+DC4[d]*j;
                int8_t v = (kr==r && kc==c) ? new_val : b.cell[kr][kc];
                if (Board::owns(v, p))  n_own++;
                else if (v == 0)        { gr=kr; gc=kc; }
                else                    n_opp++;
            }
            if (n_own==5 && n_opp==0 && gr>=0) {
                auto gap = std::make_pair(gr, gc);
                if (std::find(gaps.begin(), gaps.end(), gap) == gaps.end())
                    gaps.push_back(gap);
            }
        }
    }
    return gaps;
}

// True if player p can make 6-in-a-row THIS move (fill a gap, or strong-capture
// the single opp regular that completes a 6). Used as a VCF soundness guard:
// a forced "block" is only forced if the defender has no faster winning reply.
static bool can_win_now(const Board& b, int p) {
    bool has_str = b.strong[(p > 0) ? 0 : 1] > 0;
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            for (int d = 0; d < 4; d++) {
                int er = r + DR4[d]*5, ec = c + DC4[d]*5;
                if (er < 0 || er >= N || ec < 0 || ec >= N) continue;
                int n_own = 0, n_empty = 0, n_oppreg = 0, n_oppstr = 0;
                for (int j = 0; j < 6; j++) {
                    int8_t v = b.cell[r + DR4[d]*j][c + DC4[d]*j];
                    if (v == 0) n_empty++;
                    else if (Board::owns(v, p)) n_own++;
                    else if (std::abs((int)v) == 1) n_oppreg++;
                    else n_oppstr++;
                }
                if (n_own == 5) {
                    if (n_empty == 1) return true;                       // fill gap → 6
                    if (has_str && n_oppreg == 1 && n_oppstr == 0) return true; // capture → 6
                }
            }
        }
    }
    return false;
}

// Rush-defense: opponent is building a line through our SINGLE, capturable
// regular block. If both sides hold a strong piece, that block is illusory —
// the opponent will strong-capture it (the loss pattern the offense-tuned eval
// underrates, beyond the search horizon). Find the most-developed such 6-window
// (opp >= 3 own; the only non-opponent cell is our one regular; no own strong)
// and return the move that upgrades OUR regular there to strong (uncapturable).
// Returns the strong-move index, or -1. Defense-only — does not touch the eval.
static int find_block_upgrade(const Board& b) {
    int me = b.player, opp = -me;
    if (b.strong[(me  > 0) ? 0 : 1] <= 0) return -1;  // we need strong to upgrade
    if (b.strong[(opp > 0) ? 0 : 1] <= 0) return -1;  // opp needs strong to capture
    int best_r = -1, best_c = -1, best_opp = 2;        // require opp_own >= 3
    for (int r = 0; r < N; r++)
    for (int c = 0; c < N; c++)
    for (int d = 0; d < 4; d++) {
        int er = r + DR4[d]*5, ec = c + DC4[d]*5;
        if (er < 0 || er >= N || ec < 0 || ec >= N) continue;
        int n_opp = 0, my_reg = 0, my_str = 0, mr = -1, mc = -1;
        for (int j = 0; j < 6; j++) {
            int rr = r + DR4[d]*j, cc = c + DC4[d]*j;
            int8_t v = b.cell[rr][cc];
            if (v == 0) continue;
            if (Board::owns(v, opp)) n_opp++;
            else if (std::abs((int)v) == 2) my_str++;       // own strong already
            else { my_reg++; mr = rr; mc = cc; }            // own regular block
        }
        // window = opp pieces + exactly one of our regulars + empties (no own strong)
        if (my_str != 0 || my_reg != 1) continue;
        if (n_opp > best_opp) { best_opp = n_opp; best_r = mr; best_c = mc; }
    }
    if (best_r < 0) return -1;
    return NN + best_r*N + best_c;  // upgrade our regular → strong
}

static int vcf_rec(const Board& b, int attacker, int ply) {
    if (ply >= VCF_MAX_PLY) return -1;

    // Regular moves: empty cells that create a 5-threat
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            if (b.cell[r][c] != 0) continue;
            auto gaps = vcf_gaps(b, r, c, attacker, (int8_t)attacker);
            if (gaps.empty()) continue;

            Board after_atk = b.apply(r, c, false);
            int atk_idx = r*N + c;

            // Soundness: if defender can win immediately, they ignore our threat
            // and win first → this line is not a forced win.
            if (can_win_now(after_atk, -attacker)) continue;

            if ((int)gaps.size() >= 2) return atk_idx;  // dual threat = unblockable

            auto [gr, gc] = gaps[0];
            if (after_atk.cell[gr][gc] != 0) continue;  // gap already occupied

            Board after_blk = after_atk.apply(gr, gc, false);
            if (vcf_rec(after_blk, attacker, ply + 2) >= 0)
                return atk_idx;
        }
    }

    // Strong capture moves: capture opp regular piece to create 5-threat
    if (b.strong[b.player > 0 ? 0 : 1] > 0) {
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                if (!b.opp_reg(r, c)) continue;
                int8_t nv = (int8_t)(attacker > 0 ? 2 : -2);
                auto gaps = vcf_gaps(b, r, c, attacker, nv);
                if (gaps.empty()) continue;

                Board after_atk = b.apply(r, c, true);
                int atk_idx = NN + r*N + c;

                if (can_win_now(after_atk, -attacker)) continue;

                if ((int)gaps.size() >= 2) return atk_idx;

                auto [gr, gc] = gaps[0];
                if (after_atk.cell[gr][gc] != 0) continue;

                Board after_blk = after_atk.apply(gr, gc, false);
                if (vcf_rec(after_blk, attacker, ply + 2) >= 0)
                    return atk_idx;
            }
        }
    }

    return -1;
}

static int vcf_search(const Board& b) { return vcf_rec(b, b.player, 0); }

// ── VCT (Victory by Consecutive Threats) ──────────────────────────────────────
// Extends VCF to handle 4-threats (live fours: 4 own + 2 empty + 0 opp in window).
// Called after vcf_search fails. Uses the shared time_up_flag / search_deadline.

static constexpr int VCT_MAX_PLY = 16;  // 8 attacker+defender pairs

// After placing new_val at (r,c) for player p, return empty cells in windows
// that have exactly 4 own + 2 empty + 0 opp (live-four blocking cells).
static std::vector<std::pair<int,int>> vct_4threat_blocks(
    const Board& b, int r, int c, int p, int8_t new_val)
{
    std::vector<std::pair<int,int>> blocks;
    for (int d = 0; d < 4; d++) {
        for (int k = 0; k < 6; k++) {
            int sr = r - DR4[d]*k, sc = c - DC4[d]*k;
            int er = sr + DR4[d]*5, ec = sc + DC4[d]*5;
            if (sr<0||sr>=N||sc<0||sc>=N||er<0||er>=N||ec<0||ec>=N) continue;
            int n_own=0, n_opp=0;
            std::vector<std::pair<int,int>> empties;
            for (int j = 0; j < 6; j++) {
                int kr = sr+DR4[d]*j, kc = sc+DC4[d]*j;
                int8_t v = (kr==r && kc==c) ? new_val : b.cell[kr][kc];
                if (Board::owns(v, p)) n_own++;
                else if (v == 0) empties.push_back({kr, kc});
                else n_opp++;
            }
            if (n_own == 4 && n_opp == 0 && (int)empties.size() == 2) {
                for (auto& e : empties) {
                    if (std::find(blocks.begin(), blocks.end(), e) == blocks.end())
                        blocks.push_back(e);
                }
            }
        }
    }
    return blocks;
}

static int vct_rec(const Board& b, int attacker, int ply) {
    if (time_up_flag || ply >= VCT_MAX_PLY) return -1;

    // First try VCF (pure 5-threat chain, fully forced and cheap)
    int vcf = vcf_rec(b, attacker, ply);
    if (vcf >= 0) return vcf;

    // Generate candidates: empty cells within radius 3 of any piece
    bool visited[N][N] = {};
    std::vector<std::pair<int,int>> cands;
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            if (b.cell[r][c] == 0) continue;
            for (int dr = -3; dr <= 3; dr++) {
                for (int dc = -3; dc <= 3; dc++) {
                    int nr = r+dr, nc = c+dc;
                    if (nr<0||nr>=N||nc<0||nc>=N) continue;
                    if (b.cell[nr][nc] != 0) continue;
                    if (!visited[nr][nc]) { visited[nr][nc]=true; cands.push_back({nr,nc}); }
                }
            }
        }
    }

    // Try 4-threat (live-four) moves; skip 5-threats (already covered by vcf above)
    for (auto& [r, c] : cands) {
        if (time_up_flag) return -1;
        auto gaps5 = vcf_gaps(b, r, c, attacker, (int8_t)attacker);
        if (!gaps5.empty()) continue;

        auto blocks4 = vct_4threat_blocks(b, r, c, attacker, (int8_t)attacker);
        if (blocks4.empty()) continue;

        Board after_atk = b.apply(r, c, false);
        int atk_idx = r*N + c;

        // Win only if attacker wins against EVERY possible defender block
        bool wins_all = true;
        for (auto& [br, bc] : blocks4) {
            if (time_up_flag) return -1;
            if (after_atk.cell[br][bc] != 0) continue;
            Board after_blk = after_atk.apply(br, bc, false);
            if (vct_rec(after_blk, attacker, ply + 2) < 0) { wins_all = false; break; }
        }
        if (wins_all) return atk_idx;
    }

    return -1;
}

static int vct_search(const Board& b) { return vct_rec(b, b.player, 0); }

// ── Alpha-beta negamax with TT + History ─────────────────────────────────────

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

    for (int mi = 0; mi < (int)moves.size(); mi++) {
        if (time_up_flag) break;
        auto& [sc, m] = moves[mi];
        Board nx  = b.apply(m.r, m.c, m.is_str);

        // LMR: late moves (ranked 4+) at depth>=3 get depth reduced by 1-2.
        // Skip reduction for: captures (strong on opp-reg), high-scored (fork/4+).
        int val;
        bool is_capture  = m.is_str && b.opp_reg(m.r, m.c);
        bool is_critical = (sc >= S_4);
        // Threat extension: extend by 1 ply for critical/capture moves near horizon.
        // ply<=14 cap prevents cascading extensions on already-deep paths.
        int ext = ((is_critical || is_capture) && depth <= 2 && ply <= 14) ? 1 : 0;
        if (depth >= 3 && mi >= 4 && !is_capture && !is_critical) {
            int R = (mi >= 8) ? 2 : 1;
            val = -negamax(nx, depth - 1 - R, ply + 1, -alpha - 1, -alpha);
            if (!time_up_flag && val > alpha)
                val = -negamax(nx, depth - 1, ply + 1, -beta, -alpha);
        } else {
            val = -negamax(nx, depth - 1 + ext, ply + 1, -beta, -alpha);
        }
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
        int vcf_idx = vcf_search(b);
        if (vcf_idx >= 0) return vcf_idx;
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

        // VCF pre-search: if forced win sequence exists, return immediately
        {
            int vcf_idx = vcf_search(b);
            if (vcf_idx >= 0) return vcf_idx;
        }

        // Opponent-VCF defense: a forced opponent win can be a deep (8-14 ply)
        // forcing sequence the depth-limited IDA can't see (horizon). VCF detects
        // it cheaply. If found, play the highest-ranked move that breaks ALL of
        // the opponent's forced wins, rather than letting IDA wander into a loss.
        {
            Board ov = b;
            ov.player = -b.player;                      // pretend opponent to move
            if (vcf_rec(ov, -b.player, 0) >= 0) {
                auto cands = gen_moves(b);              // sorted best-first
                // Prefer captures: breaking a line is stronger than blocking one end.
                // (Blocking one end of double-open 4-in-a-row leaves mirror threat alive.)
                for (auto& [sc, m] : cands) {
                    if (!m.is_str || !b.opp_reg(m.r, m.c)) continue;
                    Board after = b.apply(m.r, m.c, m.is_str);
                    if (vcf_rec(after, after.player, 0) >= 0) continue;
                    return NN + m.r*N + m.c;
                }
                // Fall back: any move that breaks opp VCF
                for (auto& [sc, m] : cands) {
                    Board after = b.apply(m.r, m.c, m.is_str);  // toggles to opponent
                    if (vcf_rec(after, after.player, 0) >= 0) continue;  // still loses
                    return m.is_str ? (NN + m.r*N + m.c) : (m.r*N + m.c);
                }
                // No refutation exists → genuinely lost; fall through to IDA.
            }
        }

        // Rush-defense: upgrade an illusory (capturable) block when the opponent
        // is rushing a line through it. Defense-only pre-check.
        {
            int up = find_block_upgrade(b);
            if (up >= 0) return up;
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
        compute_initial_scores(b);
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
