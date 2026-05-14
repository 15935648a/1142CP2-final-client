"""Build C++ game engine extension: python setup_cpp.py build_ext --inplace"""
from setuptools import setup, Extension
import pybind11

ext = Extension(
    "connect6s_cpp",
    sources=[
        "connect6s/cpp/bindings.cpp",
        # game.cpp is #included in bindings.cpp to avoid a separate TU
    ],
    include_dirs=[
        pybind11.get_include(),
        "connect6s",            # for game.hpp / game.cpp
    ],
    extra_compile_args=["-O3", "-std=c++17", "-march=native"],
    language="c++",
)

setup(
    name="connect6s_cpp",
    ext_modules=[ext],
)
