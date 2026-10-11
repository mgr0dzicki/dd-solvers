"""Profiling variant of benchmark_batched_gemv.py restricted to the PyTorch baselines.

Meant to be run under Nsight Systems, e.g.:

    nsys profile --trace=cuda,nvtx,cublas --sample=none --cpuctxsw=none \
        -o ../results/profile_batched_gemv_torch python profile_batched_gemv_torch.py

Each (k, dtype, algorithm) configuration is wrapped in NVTX ranges so that the
kernels and cuBLAS calls issued by eager and compiled PyTorch can be attributed.
"""

import torch
import torch.cuda.nvtx as nvtx
import gc


def mul_torch(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (A @ b[:, :, None].to(A.dtype)).to(b.dtype)


def get_algorithms():
    torch._dynamo.reset()
    return {
        "torch": mul_torch,
        "torch_compile": torch.compile(mul_torch, dynamic=False, fullgraph=True),
    }


MAX_ELEMENTS = 2**27
WARMUP_ITERS = 2
PROFILED_ITERS = 3

ks = range(2, 100)

for k in ks:
    n = MAX_ELEMENTS // (k * k)
    for b_dtype in [torch.float32, torch.float64]:
        for A_dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            if A_dtype is torch.float64 and b_dtype is torch.float32:
                continue  # skip unsupported combo

            gc.collect()
            torch.cuda.empty_cache()
            A = (
                torch.randn((n, k, k), device="cuda", dtype=A_dtype)
                .permute(0, 2, 1)
                .contiguous()
                .permute(0, 2, 1)
            )  # column-major
            b = torch.randn((n, k), device="cuda", dtype=b_dtype)
            A_name = str(A_dtype).split(".")[-1]
            b_name = str(b_dtype).split(".")[-1]
            torch.cuda.synchronize()

            for alg_name, alg in get_algorithms().items():
                tag = f"{alg_name} k={k} A={A_name} b={b_name}"
                print(tag, flush=True)
                # For torch_compile, the first call triggers compilation.
                with nvtx.range(f"first_call {tag}"):
                    alg(A, b)
                    torch.cuda.synchronize()
                for _ in range(WARMUP_ITERS):
                    alg(A, b)
                torch.cuda.synchronize()
                with nvtx.range(f"run {tag}"):
                    for _ in range(PROFILED_ITERS):
                        alg(A, b)
                    torch.cuda.synchronize()
