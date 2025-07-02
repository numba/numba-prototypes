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

# # Demo 2: CUDA Backend for Tanh Approximation in GELU Activation Layer
#
# (Depends on Ch.08)
#
# This demo notebook shows how to use a CUDA backend to accelerate the GELU
# activation function using a Pade44 rational approximation for tanh. We build
# on the previous demo and show how to offload the computation to the GPU using
# Numba and a custom backend.
#
# The notebook demonstrates:
# - How to configure and use a GPU backend for vectorized ufuncs
# - How to run and test the optimized GELU function on CUDA
# - How to compare results with the original NumPy implementation

from numba import cuda

from ch08_gpu_offload import GPUBackend
from ch08_gpu_offload import gpu_compiler_config as _ch08_gpu_compiler_config
from demo01_gelu_tanh_approx import *
from utils.report import Report

# ## Setup GPU Backend
#
# Define a backend that combines the ufunc backend and GPU backend, enabling
# compilation and execution of vectorized functions on CUDA devices.


class GpuUfuncBackend(Backend, GPUBackend):
    # Ufunc + GPU backend
    def __init__(self, compile_only: bool = False):
        GPUBackend.__init__(self, compile_only)


gpu_compiler_config = {
    **_ch08_gpu_compiler_config,
    "converter_class": ExtendEGraphToRVSDG,
    "cost_model": MyCostModel(),
    "backend": GpuUfuncBackend(compile_only=not cuda.is_available()),
}

# ## Configure the CUDA Ufunc Pipeline
#
# Set up the pipeline to compile the GELU function as a CUDA-accelerated
# vectorized ufunc, using the GPU backend and the Pade44 tanh approximation.

report = Report("Pipeline execution report", enable_nested_metadata=True)
cuda_vectorized_gelu = ufunc_vectorize(
    input_type=Float32,
    ndim=1,
    compiler_config={**gpu_compiler_config, "pipeline_report": report, "pipeline_debug": True},
    extra_ruleset=additional_rules | optimize_rules,
)(gelu_tanh_forward)
if __name__ == "__main__":
    report.display()


# ## Test GELU Ufunc on CUDA
#
# Run the compiled CUDA ufunc on a random input and compare the result to the
# original NumPy implementation. If CUDA is unavailable, skip the test.

if __name__ == "__main__":
    if not cuda.is_available():
        print("SKIPPED. CUDA unavailable")
    else:
        relclose = lambda x, y: np.allclose(x, y, rtol=1e-6)
        input_val = np.random.random(100).astype(np.float32)
        report.display()
        run_test(
            gelu_tanh_forward,
            cuda_vectorized_gelu,
            (input_val,),
            equal=relclose,
            verbose=True,
        )

# ## Benchmark

if __name__ == "__main__":
    if not cuda.is_available():
        print("SKIPPED. CUDA unavailable")
    else:
        input_val = np.random.random(300000).astype(np.float32)
        out = np.zeros_like(input_val)
    
        print("original")
        # %timeit gelu_tanh_forward(input_val)
        print("superoptimized")
        # %timeit cuda_vectorized_gelu(input_val, out=out)
