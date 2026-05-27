// C++ MCTS — drop-in replacement for connect6s.mcts.MCTS
//
// Mirrors the Python MCTS exactly:
//   - PUCT selection
//   - Batched leaf evaluation (calls network.predict_batch / network.predict)
//   - Virtual loss (parallel-safe batch selection)
//   - Zobrist transposition cache (skip NN for seen positions)
//
// Value convention: every node stores values from the POV of the player to
// move AT that node. UCB uses q = -child.mean_value. Backprop negates value
// at every edge going up the tree.
//
// Build: see scripts/build_mcts.sh (raw g++, no setup.py).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "game.hpp"
#include "game.cpp"   // compiled once via this TU

namespace py = pybind11;
using namespace py::literals;

static constexpr float EPS = 1e-8f;
static constexpr int   VIRTUAL_LOSS = 1;

// ── Policy buffer type ────────────────────────────────────────────────────────
using PolicyArr = std::array<float, ACTION_SIZE>;

// ── _max_runs ─────────────────────────────────────────────────────────────────
// For each (r,c), max consecutive run of `board` pieces passing through (r,c)
// across 4 directions (pieces on both sides, not counting (r,c) itself).
// `board` is a bool grid (BOARD_SIZE x BOARD_SIZE). Returns int grid.
static void max_runs(const std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE>& b,
                     std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE>& best) {
    const int N = BOARD_SIZE;
    std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE> fwd{}, bwd{};

    for (auto& r : best) r.fill(0);

    auto reset = [&]() {
        for (auto& r : fwd) r.fill(0);
        for (auto& r : bwd) r.fill(0);
    };

    // Horizontal (0,1)
    reset();
    for (int r = 0; r < N; ++r) {
        for (int c = N - 2; c >= 0; --c)
            fwd[r][c] = b[r][c + 1] * (fwd[r][c + 1] + 1);
        for (int c = 1; c < N; ++c)
            bwd[r][c] = b[r][c - 1] * (bwd[r][c - 1] + 1);
    }
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            best[r][c] = std::max(best[r][c], fwd[r][c] + bwd[r][c]);

    // Vertical (1,0)
    reset();
    for (int c = 0; c < N; ++c) {
        for (int r = N - 2; r >= 0; --r)
            fwd[r][c] = b[r + 1][c] * (fwd[r + 1][c] + 1);
        for (int r = 1; r < N; ++r)
            bwd[r][c] = b[r - 1][c] * (bwd[r - 1][c] + 1);
    }
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            best[r][c] = std::max(best[r][c], fwd[r][c] + bwd[r][c]);

    // Diagonal (1,1)
    reset();
    for (int r = N - 2; r >= 0; --r)
        for (int c = N - 2; c >= 0; --c)
            fwd[r][c] = b[r + 1][c + 1] * (fwd[r + 1][c + 1] + 1);
    for (int r = 1; r < N; ++r)
        for (int c = 1; c < N; ++c)
            bwd[r][c] = b[r - 1][c - 1] * (bwd[r - 1][c - 1] + 1);
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            best[r][c] = std::max(best[r][c], fwd[r][c] + bwd[r][c]);

    // Anti-diagonal (1,-1)
    reset();
    for (int r = N - 2; r >= 0; --r)
        for (int c = 1; c < N; ++c)
            fwd[r][c] = b[r + 1][c - 1] * (fwd[r + 1][c - 1] + 1);
    for (int r = 1; r < N; ++r)
        for (int c = N - 2; c >= 0; --c)
            bwd[r][c] = b[r - 1][c + 1] * (bwd[r - 1][c + 1] + 1);
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            best[r][c] = std::max(best[r][c], fwd[r][c] + bwd[r][c]);
}

// ── _threat_prior ─────────────────────────────────────────────────────────────
// obs is the (6,15,15) observation. own = obs[0]+obs[1] > 0.5, opp = obs[2]+obs[3].
// scores[idx] = own_runs[r,c] + 3.0 * opp_runs[r,c] for each legal move.
static void threat_prior(const GameState::Obs& obs,
                         const std::vector<Move>& legal,
                         PolicyArr& scores) {
    const int N = BOARD_SIZE;
    std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE> own{}, opp{};
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            own[r][c] = (obs[0][r][c] + obs[1][r][c]) > 0.5f ? 1 : 0;
            opp[r][c] = (obs[2][r][c] + obs[3][r][c]) > 0.5f ? 1 : 0;
        }
    std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE> own_runs{}, opp_runs{};
    max_runs(own, own_runs);
    max_runs(opp, opp_runs);

    scores.fill(0.0f);
    for (const auto& mv : legal) {
        int idx = (mv.is_strong ? N * N : 0) + mv.row * N + mv.col;
        scores[idx] = float(own_runs[mv.row][mv.col]) +
                      3.0f * float(opp_runs[mv.row][mv.col]);
    }
}

// ── MCTS Node ─────────────────────────────────────────────────────────────────
struct MCTSNode {
    GameState state;
    float     prior{0.0f};
    int32_t   visit_count{0};
    float     value_sum{0.0f};
    int32_t   in_flight{0};
    std::unordered_map<int, int32_t> children;  // action_idx -> pool_idx

    explicit MCTSNode(const GameState& s, float p = 0.0f) : state(s), prior(p) {}

    float mean_value() const {
        return (value_sum + float(in_flight)) / (float(visit_count) + EPS);
    }
    float ucb_score(int parent_n, float c_puct) const {
        float q = -mean_value();
        float u = c_puct * prior * std::sqrt(float(parent_n)) /
                  (1.0f + float(visit_count));
        return q + u;
    }
    bool is_leaf() const { return children.empty(); }
    void add_virtual_loss() {
        in_flight   += VIRTUAL_LOSS;
        visit_count += VIRTUAL_LOSS;
    }
    void revert_virtual_loss() {
        in_flight   -= VIRTUAL_LOSS;
        visit_count -= VIRTUAL_LOSS;
    }
};

// ── MCTS ──────────────────────────────────────────────────────────────────────
class MCTSCpp {
public:
    MCTSCpp(py::object network, int num_simulations = 400, float c_puct = 1.5f,
            float dirichlet_alpha = 0.3f, float dirichlet_eps = 0.25f,
            int leaf_batch_size = 16, float heuristic_weight = 0.0f)
        : network_(std::move(network)),
          num_simulations_(num_simulations),
          c_puct_(c_puct),
          dirichlet_alpha_(dirichlet_alpha),
          dirichlet_eps_(dirichlet_eps),
          leaf_batch_size_(leaf_batch_size),
          heuristic_weight_(heuristic_weight),
          rng_(std::random_device{}()) {}

    void clear_cache() { cache_.clear(); }

    // ── run() ──────────────────────────────────────────────────────────────────
    // Returns a numpy visit-count vector of shape [ACTION_SIZE].
    // deadline is epoch seconds (py::none() means no deadline).
    py::array_t<float> run(const GameState& state, bool add_noise = true,
                           py::object deadline = py::none()) {
        bool   has_deadline = !deadline.is_none();
        double deadline_s   = has_deadline ? deadline.cast<double>() : 0.0;

        pool_.clear();
        int root_idx = alloc_node(state);
        ensure_cached(root_idx);
        if (add_noise) add_dirichlet_noise(root_idx);
        filter_forced_block(root_idx);  // restrict to blocking moves if 5-threat exists

        int done = 0;
        const int B = leaf_batch_size_;

        while (done < num_simulations_ &&
               (!has_deadline || now_seconds() < deadline_s)) {
            int batch = std::min(B, num_simulations_ - done);

            // ── SELECT (with virtual loss) ─────────────────────────────────────
            std::vector<std::pair<int, std::vector<int>>> selected;
            selected.reserve(batch);
            for (int b = 0; b < batch; ++b) {
                int node_idx = root_idx;
                std::vector<int> path;
                path.push_back(node_idx);
                pool_[node_idx].add_virtual_loss();
                while (!pool_[node_idx].is_leaf() &&
                       !pool_[node_idx].state.game_over) {
                    int child_idx = best_child(node_idx);
                    pool_[child_idx].add_virtual_loss();
                    node_idx = child_idx;
                    path.push_back(node_idx);
                }
                selected.emplace_back(node_idx, std::move(path));
            }

            // ── BATCH EVAL: collect unique un-cached non-terminal leaves ───────
            std::unordered_map<uint64_t, int> to_eval;  // zobrist -> node_idx
            for (auto& sel : selected) {
                int leaf_idx = sel.first;
                if (pool_[leaf_idx].state.game_over) continue;
                uint64_t key = pool_[leaf_idx].state.zobrist_hash;
                if (cache_.find(key) == cache_.end() &&
                    to_eval.find(key) == to_eval.end()) {
                    to_eval[key] = leaf_idx;
                }
            }

            if (!to_eval.empty()) {
                evaluate_batch(to_eval);
            }

            // ── EXPAND + BACKPROP ─────────────────────────────────────────────
            for (auto& sel : selected) {
                int leaf_idx = sel.first;
                auto& path = sel.second;

                // Revert virtual losses first.
                for (int nidx : path)
                    pool_[nidx].revert_virtual_loss();

                float value;
                if (pool_[leaf_idx].state.game_over) {
                    value = (pool_[leaf_idx].state.winner != 0) ? -1.0f : 0.0f;
                } else {
                    uint64_t key = pool_[leaf_idx].state.zobrist_hash;
                    auto it = cache_.find(key);
                    const PolicyArr& masked = it->second.first;
                    value = it->second.second;
                    if (pool_[leaf_idx].is_leaf()) {
                        expand(leaf_idx, masked);
                    }
                }

                float v = value;
                for (auto rit = path.rbegin(); rit != path.rend(); ++rit) {
                    MCTSNode& nd = pool_[*rit];
                    nd.visit_count += 1;
                    nd.value_sum   += v;
                    v = -v;
                }
            }

            done += batch;
        }

        // ── Build visit-count vector ───────────────────────────────────────────
        py::array_t<float> counts(ACTION_SIZE);
        auto buf = counts.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < ACTION_SIZE; ++i) buf(i) = 0.0f;
        for (const auto& kv : pool_[root_idx].children) {
            int action_idx = kv.first;
            int child_idx  = kv.second;
            buf(action_idx) = float(pool_[child_idx].visit_count);
        }
        return counts;
    }

    // ── get_action_probs_timed ─────────────────────────────────────────────────
    py::array_t<float> get_action_probs_timed(const GameState& state,
                                              double seconds,
                                              bool add_noise = true) {
        double deadline = now_seconds() + seconds;
        int saved = num_simulations_;
        num_simulations_ = 10000000;
        py::array_t<float> counts;
        try {
            counts = run(state, add_noise, py::float_(deadline));
        } catch (...) {
            num_simulations_ = saved;
            throw;
        }
        num_simulations_ = saved;
        return greedy_probs(counts);
    }

    // ── get_action_probs ───────────────────────────────────────────────────────
    py::array_t<float> get_action_probs(const GameState& state,
                                        double temperature = 1.0,
                                        bool add_noise = true) {
        py::array_t<float> counts = run(state, add_noise, py::none());
        auto cbuf = counts.unchecked<1>();

        if (temperature == 0.0) {
            return greedy_probs(counts);
        }

        std::array<double, ACTION_SIZE> powered{};
        double s = 0.0;
        double inv_t = 1.0 / temperature;
        for (py::ssize_t i = 0; i < ACTION_SIZE; ++i) {
            double v = std::pow(double(cbuf(i)), inv_t);
            powered[i] = v;
            s += v;
        }

        py::array_t<float> probs(ACTION_SIZE);
        auto pbuf = probs.mutable_unchecked<1>();

        if (s < EPS) {
            for (py::ssize_t i = 0; i < ACTION_SIZE; ++i) pbuf(i) = 0.0f;
            auto legal = state.legal_moves();
            float w = legal.empty() ? 0.0f : 1.0f / float(legal.size());
            for (const auto& mv : legal)
                pbuf(mv.to_index()) = w;
            return probs;
        }
        for (py::ssize_t i = 0; i < ACTION_SIZE; ++i)
            pbuf(i) = float(powered[i] / s);
        return probs;
    }

    // ── select_action ──────────────────────────────────────────────────────────
    py::object select_action(const GameState& state, double temperature = 1.0,
                             bool add_noise = true) {
        py::array_t<float> probs = get_action_probs(state, temperature, add_noise);
        auto pbuf = probs.unchecked<1>();
        // Sample from probs.
        std::array<double, ACTION_SIZE> cum{};
        double total = 0.0;
        for (py::ssize_t i = 0; i < ACTION_SIZE; ++i) {
            total += double(pbuf(i));
            cum[i] = total;
        }
        std::uniform_real_distribution<double> dist(0.0, total);
        double x = dist(rng_);
        int action_idx = ACTION_SIZE - 1;
        for (int i = 0; i < ACTION_SIZE; ++i) {
            if (x <= cum[i]) { action_idx = i; break; }
        }
        Move mv = Move::from_index(action_idx);
        py::tuple action = py::make_tuple(mv.row, mv.col, mv.is_strong);
        return py::make_tuple(action, probs);
    }

    // ── Mutable attribute accessors ────────────────────────────────────────────
    int  get_num_simulations() const { return num_simulations_; }
    void set_num_simulations(int v)  { num_simulations_ = v; }
    float get_heuristic_weight() const { return heuristic_weight_; }
    void  set_heuristic_weight(float v) { heuristic_weight_ = v; }
    py::object get_network() const { return network_; }

private:
    py::object network_;
    int    num_simulations_;
    float  c_puct_;
    float  dirichlet_alpha_;
    float  dirichlet_eps_;
    int    leaf_batch_size_;
    float  heuristic_weight_;
    std::mt19937 rng_;

    std::deque<MCTSNode> pool_;  // stable references on push_back
    std::unordered_map<uint64_t, std::pair<PolicyArr, float>> cache_;

    // ── Allocation ─────────────────────────────────────────────────────────────
    int alloc_node(const GameState& s, float prior = 0.0f) {
        pool_.emplace_back(s, prior);
        return int(pool_.size()) - 1;
    }

    static double now_seconds() {
        return std::chrono::duration<double>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    // ── mask_and_normalize ─────────────────────────────────────────────────────
    static PolicyArr mask_and_normalize(const PolicyArr& policy,
                                        const std::vector<Move>& legal) {
        PolicyArr masked{};
        masked.fill(0.0f);
        for (const auto& mv : legal) {
            int idx = (mv.is_strong ? BOARD_SIZE * BOARD_SIZE : 0) +
                      mv.row * BOARD_SIZE + mv.col;
            masked[idx] = policy[idx];
        }
        float s = 0.0f;
        for (float v : masked) s += v;
        if (s > EPS) {
            for (float& v : masked) v /= s;
        } else {
            float w = 1.0f / float(legal.size());
            for (const auto& mv : legal) {
                int idx = (mv.is_strong ? BOARD_SIZE * BOARD_SIZE : 0) +
                          mv.row * BOARD_SIZE + mv.col;
                masked[idx] = w;
            }
        }
        return masked;
    }

    // ── filter_forced_block ────────────────────────────────────────────────────
    // If opponent has a 5-in-a-row threat at root, restrict search to blocking
    // moves only. Deterministic at root only — does not affect inner search.
    // This prevents the network from "ignoring" forced blocks and ensures the
    // policy target in training is one-hot on the correct blocking cell.
    void filter_forced_block(int root_idx) {
        MCTSNode& root = pool_[root_idx];
        if (root.children.empty()) return;

        GameState::Obs obs = root.state.observation();
        const int N = BOARD_SIZE;

        std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE> opp_board{};
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c)
                opp_board[r][c] = (obs[2][r][c] + obs[3][r][c]) > 0.5f ? 1 : 0;

        std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE> opp_runs{};
        max_runs(opp_board, opp_runs);

        int max_threat = 0;
        for (const auto& kv : root.children) {
            int flat = kv.first % (N * N);
            max_threat = std::max(max_threat, opp_runs[flat / N][flat % N]);
        }

        // Block 4-in-a-row too: opponent 2 moves from win, must respond now
        if (max_threat >= 4) {
            std::vector<int> to_remove;
            for (const auto& kv : root.children) {
                int flat = kv.first % (N * N);
                if (opp_runs[flat / N][flat % N] < 4)
                    to_remove.push_back(kv.first);
            }
            for (int idx : to_remove)
                root.children.erase(idx);
        }
    }

    // ── best_child ─────────────────────────────────────────────────────────────
    int best_child(int node_idx) {
        const MCTSNode& node = pool_[node_idx];
        int parent_n = node.visit_count;
        // FPU: unvisited children use parent's Q instead of 0.
        // Prevents Q=0 optimism from beating visited nodes with negative Q.
        float fpu = node.mean_value();
        float best_s = -std::numeric_limits<float>::infinity();
        int   best_i = -1;
        for (const auto& kv : node.children) {
            int child_idx = kv.second;
            const MCTSNode& child = pool_[child_idx];
            float s;
            if (child.visit_count + child.in_flight == 0) {
                s = fpu + c_puct_ * child.prior * std::sqrt(float(parent_n));
            } else {
                s = child.ucb_score(parent_n, c_puct_);
            }
            if (s > best_s) {
                best_s = s;
                best_i = child_idx;
            }
        }
        return best_i;
    }

    // ── expand ─────────────────────────────────────────────────────────────────
    void expand(int node_idx, const PolicyArr& masked) {
        // Snapshot state because alloc_node may reallocate deque internals
        // (deque keeps element refs stable, but be defensive on the copy).
        GameState parent_state = pool_[node_idx].state;
        auto legal = parent_state.legal_moves();
        for (const auto& mv : legal) {
            int idx = mv.to_index();
            GameState child_state = parent_state.apply(mv);
            int child_idx = alloc_node(child_state, masked[idx]);
            pool_[node_idx].children[idx] = child_idx;
        }
    }

    // ── ensure_cached ──────────────────────────────────────────────────────────
    // Single-eval path used for the root. Caches policy/value and expands leaf.
    float ensure_cached(int node_idx) {
        GameState state = pool_[node_idx].state;
        uint64_t key = state.zobrist_hash;

        if (cache_.find(key) == cache_.end()) {
            if (state.game_over) {
                PolicyArr zeros{}; zeros.fill(0.0f);
                cache_[key] = {zeros, 0.0f};
            } else {
                auto legal = state.legal_moves();
                if (legal.empty()) {
                    PolicyArr zeros{}; zeros.fill(0.0f);
                    cache_[key] = {zeros, 0.0f};
                } else {
                    GameState::Obs obs = state.observation();
                    PolicyArr policy;
                    float value;
                    single_predict(obs, policy, value);
                    if (heuristic_weight_ > 0.0f) {
                        PolicyArr threat;
                        threat_prior(obs, legal, threat);
                        for (int i = 0; i < ACTION_SIZE; ++i) {
                            float e = std::clamp(threat[i] * heuristic_weight_,
                                                 0.0f, 15.0f);
                            policy[i] *= std::exp(e);
                        }
                    }
                    PolicyArr masked = mask_and_normalize(policy, legal);
                    cache_[key] = {masked, value};
                }
            }
        }

        const auto& entry = cache_[key];
        float value = entry.second;
        if (pool_[node_idx].is_leaf() && !state.game_over) {
            auto legal = state.legal_moves();
            if (!legal.empty()) {
                expand(node_idx, entry.first);
            }
        }
        return value;
    }

    // ── single_predict ─────────────────────────────────────────────────────────
    // Calls network.predict(obs) -> (policy[450], value float).
    void single_predict(const GameState::Obs& obs, PolicyArr& policy,
                        float& value) {
        py::array_t<float> arr({6, BOARD_SIZE, BOARD_SIZE});
        auto buf = arr.mutable_unchecked<3>();
        for (int ch = 0; ch < 6; ++ch)
            for (int r = 0; r < BOARD_SIZE; ++r)
                for (int c = 0; c < BOARD_SIZE; ++c)
                    buf(ch, r, c) = obs[ch][r][c];

        py::object result = network_.attr("predict")(arr);
        py::tuple t = result.cast<py::tuple>();
        py::array_t<float> pol = py::array_t<float, py::array::c_style |
                                                    py::array::forcecast>(t[0]);
        value = t[1].cast<float>();
        auto pbuf = pol.unchecked<1>();
        for (int i = 0; i < ACTION_SIZE; ++i)
            policy[i] = pbuf(i);
    }

    // ── evaluate_batch ─────────────────────────────────────────────────────────
    // For all nodes in to_eval, build obs batch, call predict_batch, cache.
    void evaluate_batch(const std::unordered_map<uint64_t, int>& to_eval) {
        int n = int(to_eval.size());
        std::vector<int>      node_indices;
        std::vector<uint64_t> keys;
        node_indices.reserve(n);
        keys.reserve(n);
        for (const auto& kv : to_eval) {
            keys.push_back(kv.first);
            node_indices.push_back(kv.second);
        }

        py::array_t<float> obs_arr({n, 6, BOARD_SIZE, BOARD_SIZE});
        auto obuf = obs_arr.mutable_unchecked<4>();
        std::vector<GameState::Obs> obs_list(n);
        for (int i = 0; i < n; ++i) {
            obs_list[i] = pool_[node_indices[i]].state.observation();
            const auto& obs = obs_list[i];
            for (int ch = 0; ch < 6; ++ch)
                for (int r = 0; r < BOARD_SIZE; ++r)
                    for (int c = 0; c < BOARD_SIZE; ++c)
                        obuf(i, ch, r, c) = obs[ch][r][c];
        }

        py::object result = network_.attr("predict_batch")(obs_arr);
        py::tuple t = result.cast<py::tuple>();
        py::array_t<float> policies =
            py::array_t<float, py::array::c_style | py::array::forcecast>(t[0]);
        py::array_t<float> values =
            py::array_t<float, py::array::c_style | py::array::forcecast>(t[1]);
        auto pbuf = policies.unchecked<2>();   // [n, 450]
        auto vbuf = values.unchecked<1>();     // [n]

        for (int i = 0; i < n; ++i) {
            auto legal = pool_[node_indices[i]].state.legal_moves();
            PolicyArr masked{};
            if (!legal.empty()) {
                PolicyArr pol;
                for (int j = 0; j < ACTION_SIZE; ++j)
                    pol[j] = pbuf(i, j);
                if (heuristic_weight_ > 0.0f) {
                    PolicyArr threat;
                    threat_prior(obs_list[i], legal, threat);
                    for (int j = 0; j < ACTION_SIZE; ++j) {
                        float e = std::clamp(threat[j] * heuristic_weight_,
                                             0.0f, 15.0f);
                        pol[j] *= std::exp(e);
                    }
                }
                masked = mask_and_normalize(pol, legal);
            } else {
                masked.fill(0.0f);
            }
            cache_[keys[i]] = {masked, float(vbuf(i))};
        }
    }

    // ── add_dirichlet_noise ────────────────────────────────────────────────────
    void add_dirichlet_noise(int root_idx) {
        std::vector<int> child_indices;
        for (const auto& kv : pool_[root_idx].children)
            child_indices.push_back(kv.second);
        if (child_indices.empty()) return;

        // Sample Dirichlet via independent Gamma draws, then normalize.
        std::gamma_distribution<float> gamma(dirichlet_alpha_, 1.0f);
        std::vector<float> noise(child_indices.size());
        float sum = 0.0f;
        for (size_t i = 0; i < child_indices.size(); ++i) {
            float g = gamma(rng_);
            noise[i] = g;
            sum += g;
        }
        if (sum <= 0.0f) sum = EPS;
        for (size_t i = 0; i < child_indices.size(); ++i) {
            float n = noise[i] / sum;
            MCTSNode& child = pool_[child_indices[i]];
            child.prior = (1.0f - dirichlet_eps_) * child.prior +
                          dirichlet_eps_ * n;
        }
    }

    // ── greedy_probs ───────────────────────────────────────────────────────────
    static py::array_t<float> greedy_probs(const py::array_t<float>& counts) {
        auto cbuf = counts.unchecked<1>();
        py::array_t<float> probs(ACTION_SIZE);
        auto pbuf = probs.mutable_unchecked<1>();
        int best_i = 0;
        float best_v = -std::numeric_limits<float>::infinity();
        for (py::ssize_t i = 0; i < ACTION_SIZE; ++i) {
            pbuf(i) = 0.0f;
            if (cbuf(i) > best_v) { best_v = cbuf(i); best_i = int(i); }
        }
        pbuf(best_i) = 1.0f;
        return probs;
    }
};

// ── pybind11 module ───────────────────────────────────────────────────────────
PYBIND11_MODULE(connect6s_mcts_cpp, m) {
    m.doc() = "C++ MCTS for Connect-6S (drop-in for connect6s.mcts.MCTS)";
    m.attr("ACTION_SIZE") = ACTION_SIZE;
    m.attr("BOARD_SIZE")  = BOARD_SIZE;

    py::class_<MCTSCpp>(m, "MCTS")
        .def(py::init<py::object, int, float, float, float, int, float>(),
             "network"_a, "num_simulations"_a = 400, "c_puct"_a = 1.5f,
             "dirichlet_alpha"_a = 0.3f, "dirichlet_eps"_a = 0.25f,
             "leaf_batch_size"_a = 16, "heuristic_weight"_a = 0.0f)

        .def("clear_cache", &MCTSCpp::clear_cache)

        .def("run", &MCTSCpp::run,
             "state"_a, "add_noise"_a = true, "deadline"_a = py::none())

        .def("get_action_probs", &MCTSCpp::get_action_probs,
             "state"_a, "temperature"_a = 1.0, "add_noise"_a = true)

        .def("get_action_probs_timed", &MCTSCpp::get_action_probs_timed,
             "state"_a, "seconds"_a, "add_noise"_a = true)

        .def("select_action", &MCTSCpp::select_action,
             "state"_a, "temperature"_a = 1.0, "add_noise"_a = true)

        // Mutable attributes accessed by workers.
        .def_property("num_simulations",
                      &MCTSCpp::get_num_simulations,
                      &MCTSCpp::set_num_simulations)
        .def_property("heuristic_weight",
                      &MCTSCpp::get_heuristic_weight,
                      &MCTSCpp::set_heuristic_weight)
        .def_property_readonly("network", &MCTSCpp::get_network);
}
