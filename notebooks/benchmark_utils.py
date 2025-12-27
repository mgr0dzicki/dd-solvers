import torch


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
