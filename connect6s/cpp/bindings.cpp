#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../game.hpp"
#include "../game.cpp"   // compiled once via this TU

namespace py = pybind11;
using namespace py::literals;

// ── helpers ───────────────────────────────────────────────────────────────────

// Build a {1: b, -1: w} Python dict matching Python GameState.strong_pieces
static py::object make_strong_pieces(const GameState& s) {
    py::dict d;
    d[py::int_(1)]  = s.strong_pieces[0];
    d[py::int_(-1)] = s.strong_pieces[1];
    return d;
}

// Return observation as numpy (6, 15, 15) float32
static py::array_t<float> observation_np(const GameState& s) {
    auto obs    = s.observation();
    py::array_t<float> arr({6, BOARD_SIZE, BOARD_SIZE});
    auto buf = arr.mutable_unchecked<3>();
    for (int ch = 0; ch < 6; ++ch)
        for (int r = 0; r < BOARD_SIZE; ++r)
            for (int c = 0; c < BOARD_SIZE; ++c)
                buf(ch, r, c) = obs[ch][r][c];
    return arr;
}

// Return legal moves as list of (row, col, is_strong) tuples
static py::list legal_moves_py(const GameState& s) {
    auto moves = s.legal_moves();
    py::list result;
    for (const auto& mv : moves)
        result.append(py::make_tuple(mv.row, mv.col, mv.is_strong));
    return result;
}

// Return legal moves as set of tuples (for HumanAgent)
static py::set legal_move_set_py(const GameState& s) {
    auto moves = s.legal_moves();
    py::set result;
    for (const auto& mv : moves)
        result.add(py::make_tuple(mv.row, mv.col, mv.is_strong));
    return result;
}

// ── module ────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(connect6s_cpp, m) {
    m.doc() = "C++ game engine for Connect-6S (drop-in for connect6s.game.GameState)";
    m.attr("BOARD_SIZE")  = BOARD_SIZE;
    m.attr("WIN_LENGTH")  = WIN_LENGTH;
    m.attr("ACTION_SIZE") = ACTION_SIZE;

    py::class_<GameState>(m, "GameState")
        .def(py::init<>())

        // ── Properties matching Python GameState ──────────────────────────
        .def_readonly("current_player", &GameState::current_player)
        .def_readonly("move_count",     &GameState::move_count)
        .def_readonly("game_over",      &GameState::game_over)
        .def_readonly("winner",         &GameState::winner)
        .def_readonly("zobrist_hash",   &GameState::zobrist_hash)
        .def_property_readonly("strong_pieces", make_strong_pieces)

        // ── Methods ───────────────────────────────────────────────────────
        .def("copy", [](const GameState& s) { return s; })  // value copy

        .def("get_legal_moves",   legal_moves_py)
        .def("get_legal_move_set", legal_move_set_py)

        .def("action_to_index",
             [](const GameState&, int r, int c, bool is_s) {
                 return Move{r, c, is_s}.to_index();
             }, "row"_a, "col"_a, "is_strong"_a = false)

        .def("index_to_action",
             [](const GameState&, int idx) {
                 auto mv = Move::from_index(idx);
                 return py::make_tuple(mv.row, mv.col, mv.is_strong);
             })

        .def("make_move",
             [](const GameState& s, int row, int col, bool is_strong) {
                 return s.apply({row, col, is_strong});
             }, "row"_a, "col"_a, "is_strong"_a = false)

        .def("get_observation",          observation_np)
        .def("get_canonical_observation", observation_np)

        .def("render", [](const GameState& s) { s.render(); })

        .def("__repr__", [](const GameState& s) {
            return "<GameState move=" + std::to_string(s.move_count)
                 + " player=" + (s.current_player == 1 ? "Black" : "White")
                 + (s.game_over ? " OVER" : "") + ">";
        });
}
