#!/bin/bash
# Build the C++ heuristic bot extension (connect6s_heuristic_cpp).
set -e
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
PYBIND_INC=$($PYTHON -c "import pybind11; print(pybind11.get_include())")
PY_INC=$($PYTHON -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_EXT=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

g++ -O3 -march=native -ffast-math -shared -fPIC \
    -std=c++17 \
    -I"$PYBIND_INC" -I"$PY_INC" \
    connect6s/heuristic_cpp.cpp \
    -o "connect6s_heuristic_cpp${PY_EXT}"

echo "Built: connect6s_heuristic_cpp${PY_EXT}"
