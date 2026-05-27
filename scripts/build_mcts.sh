#!/bin/bash
# Build the C++ MCTS extension (connect6s_mcts_cpp) via raw g++.
# Drop-in replacement for connect6s.mcts.MCTS — see connect6s/mcts.py.
set -e
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
PYBIND_INC=$($PYTHON -c "import pybind11; print(pybind11.get_include())")
PY_INC=$($PYTHON -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_EXT=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# -DNDEBUG strips game.cpp's over-strict asserts (they assume strong moves
# only target opponent regulars; the real rule allows any regular piece).
# This matches how the existing connect6s_cpp.so was built.
g++ -O3 -march=native -ffast-math -DNDEBUG -shared -fPIC \
    -std=c++17 \
    -I"$PYBIND_INC" -I"$PY_INC" \
    -Iconnect6s \
    connect6s/mcts_cpp.cpp \
    -o "connect6s_mcts_cpp${PY_EXT}"

echo "Built: connect6s_mcts_cpp${PY_EXT}"
