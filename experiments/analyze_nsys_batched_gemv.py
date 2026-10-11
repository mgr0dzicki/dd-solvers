"""Attribute GPU kernels and cuBLAS API calls to the NVTX ranges emitted by
profile_batched_gemv_torch.py. Takes the sqlite export of the nsys report."""

import re
import sqlite3
import sys
import pandas as pd

db = sqlite3.connect(sys.argv[1])
out_csv = sys.argv[2]

strings = dict(db.execute("SELECT id, value FROM StringIds"))

nvtx = pd.read_sql(
    "SELECT start, end, text, textId, globalTid, domainId FROM NVTX_EVENTS "
    "WHERE end IS NOT NULL",
    db,
)
nvtx["text"] = nvtx["text"].fillna(nvtx["textId"].map(strings))
ranges = nvtx[nvtx.text.str.startswith("run ")].copy()
cublas = nvtx[nvtx.text.str.startswith("cublas")]

runtime = pd.read_sql(
    "SELECT start, correlationId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME", db
)
kernels = pd.read_sql(
    "SELECT correlationId, demangledName, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL",
    db,
)
kernels["name"] = kernels["demangledName"].map(strings)
kernels = kernels.merge(runtime, on="correlationId", suffixes=("", "_api"))

rows = []
for r in ranges.itertuples():
    m = re.match(r"run (\S+) k=(\d+) A=(\S+) b=(\S+)", r.text)
    alg, k, A, b = m.groups()
    in_range = (kernels.start_api >= r.start) & (kernels.start_api <= r.end)
    ks = kernels[in_range].sort_values("start_api")
    cb = cublas[(cublas.start >= r.start) & (cublas.end <= r.end)]
    n_iters = 3
    names = list(ks.name)
    per_iter = names[: len(names) // n_iters]
    rows.append(
        {
            "algorithm": alg,
            "k": int(k),
            "A_dtype": A,
            "b_dtype": b,
            "kernels_per_iter": len(per_iter),
            "kernels": " | ".join(per_iter),
            "cublas_calls": " | ".join(sorted(set(cb.text))),
            "gpu_time_ms_per_iter": (ks.end - ks.start).sum() / n_iters / 1e6,
        }
    )

df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)
print(f"wrote {len(df)} rows to {out_csv}")
