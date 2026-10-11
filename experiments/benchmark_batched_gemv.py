import torch
import dd_solvers
import pandas as pd
import tqdm
import gc
from benchmark_utils import timeit


def mul_torch(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (A @ b[:, :, None].to(A.dtype)).to(b.dtype)


def mul_custom_irrespective(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return dd_solvers.gemv_strided_batched(A, b, use_shared_memory=False)


def mul_custom_synch(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return dd_solvers.gemv_strided_batched(A, b, use_shared_memory=True)


def get_algorithms():
    # Compile from scratch for every configuration: a single shared compiled
    # function would hit Dynamo's recompile limit and silently fall back to eager
    # mode. dynamic=False specializes the generated code to the exact shapes.
    torch._dynamo.reset()
    return {
        "irrespective": mul_custom_irrespective,
        "synchronized": mul_custom_synch,
        "torch": mul_torch,
        "torch_compile": torch.compile(mul_torch, dynamic=False, fullgraph=True),
    }


MAX_ELEMENTS = 2**27

ks = range(2, 150)
results = []

for k in tqdm.tqdm(ks):
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
            exact = A.to(torch.float64) @ b.to(torch.float64)[:, :, None]

            for alg_name, alg in get_algorithms().items():
                # For torch_compile, this first call also triggers compilation,
                # so it is not included in the timings below.
                err_norm = torch.linalg.norm(
                    exact.flatten() - alg(A, b).flatten()
                ).item()
                iters = 10 if alg_name.startswith("torch") and k < 40 else 100
                times = [
                    timeit(
                        lambda: alg(A, b),
                        warmup_iters=iters // 2,
                        iters=iters,
                        repetitions=1,
                    )
                    for _ in range(10)
                ]
                results.append(
                    {
                        "n": n,
                        "k": k,
                        "A_dtype": str(A_dtype).split(".")[-1],
                        "b_dtype": str(b_dtype).split(".")[-1],
                        "algorithm": alg_name,
                        "times (ms)": times,
                        "error norm": err_norm,
                    }
                )

df = pd.DataFrame(results)
df.to_csv("../results/benchmark_batched_gemv.csv", index=False)
