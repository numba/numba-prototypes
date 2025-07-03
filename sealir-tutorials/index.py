# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# <img src="./img/book_cover_baby_dragon.jpeg" width="400px" alt="Baby Dragon Book Cover" />
#
# # Numba v2 Compiler Design <img src="https://numba.pydata.org/_static/numba-blue-icon-rgb.svg" width="80" style="float:right;"/>
#
#
# This book describes the compiler design of the next generation Numba compiler.  This compiler and its components are *extremely experimental* and under rapid development.  If you have questions, [raise an issue](https://github.com/numba/numba-prototypes/issues) on the GitHub repository.
#
# ## Chapters
#
# * [Chapter 1 - Basic Compiler Design](ch01_basic_compiler.html)
# * [Chapter 2 - EGraph Basics](ch02_egraph_basic.html)
# * [Chapter 3 - Rewriting Programs with EGraphs](ch03_egraph_program_rewrites.html)
# * Chapter 4 - Scalar Type Inference with EGraphs
#   - [Part 0 - Type inference for scalar operations](ch04_0_typeinfer_prelude.html)
#   - [Part 1 - Fully typing a scalar function with if-else branch](ch04_1_typeinfer_ifelse.html)
#   - [Part 2 - Fully typing a scalar function with loops](ch04_2_typeinfer_loops.html)
# * [Chapter 5 - Array Type Inference with EGraphs](ch05_typeinfer_array.html)
# * [Chapter 6 - MLIR Backend for Scalar Functions](ch06_mlir_backend.html)
# * [Chapter 7 - MLIR Backend for Array Functions](ch07_mlir_ufunc.html)
# * [Chapter 8 - MLIR Backend for Array Offload to the GPU](ch08_gpu_offload.html)
# * [Chapter 9 - Whole Program Compiler Driver](ch09_whole_program_compiler_driver.html)
# * Chapter 10 - Tensor Graph Extraction
# * Chapter 11 - Tensor Optimization
# * Chapter 12 - Implementing Alternative GEMMS
#
# ## Demos
# * [Demo 1 - GELU tanh approximation](demo01_gelu_tanh_approx.html)
# * [Demo 2 - GELU tanh approximation CUDA offload](demo02_cuda_ufunc.html)
# * [Demo 3.0 - Matrix Multiplication Associativity](demo03_0_matmul_assoc.html)
# * [Demo 3.1 - Tensor Optimization](demo03_1_tensor_optimization.html)
# * Demo 4: Energy Efficient GEMMs
#
