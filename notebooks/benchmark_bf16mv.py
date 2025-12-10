import torch
import dd_solvers
import pandas as pd
import tqdm
import gc


def timeit(fun, warmup_iters=10, iters=50, repetitions=3):
    times = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(warmup_iters):
            fun()

        start.record()
        for _ in range(iters):
            fun()
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end) / iters)

    return min(times)


@torch.compile
def mul_torch_bf16(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (A @ b[:, :, None].to(torch.bfloat16)).to(torch.float32)


@torch.compile
def mul_torch_fp16(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (A @ b[:, :, None].to(torch.float16)).to(torch.float32)


@torch.compile
def mul_torch(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return A @ b[:, :, None]


def mul_custom(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return dd_solvers.batch_matmul(A, b)


MAX_ELEMENTS = 2**30

ks = range(2, 130)
torch_algs = {
    torch.bfloat16: mul_torch_bf16,
    torch.float16: mul_torch_fp16,
    torch.float32: mul_torch,
    torch.float64: mul_torch,
}
dtypes = list(torch_algs.keys())
results = []

for k in tqdm.tqdm(ks):
    n = min(10_000_000, MAX_ELEMENTS // (k * k))
    for dtype in dtypes:
        dtype_bytes = torch.finfo(dtype).bits // 8
        gc.collect()
        torch.cuda.empty_cache()
        A = (
            torch.randn((n, k, k), device="cuda", dtype=dtype)
            .permute(0, 2, 1)
            .contiguous()
            .permute(0, 2, 1)
        )  # column-major
        b = torch.randn(
            (n, k),
            device="cuda",
            dtype=dtype if dtype in [torch.float32, torch.float64] else torch.float32,
        )
        exact = A.to(torch.float64) @ b.to(torch.float64)[:, :, None]

        for alg, alg_name in [
            (mul_custom, "custom"),
            (torch_algs[dtype], "torch"),
        ]:
            err_norm = torch.linalg.norm(exact.flatten() - alg(A, b).flatten()).item()
            time = timeit(lambda: alg(A, b))
            results.append(
                {
                    "n": n,
                    "k": k,
                    "dtype": str(dtype).split(".")[-1],
                    "algorithm": alg_name,
                    "time (ms)": time,
                    "error norm": err_norm,
                }
            )

df = pd.DataFrame(results)
df.to_csv("bf16mv_benchmark.csv", index=False)
