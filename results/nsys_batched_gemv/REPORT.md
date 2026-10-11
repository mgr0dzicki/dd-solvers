# What PyTorch runs for batched GEMV (eager vs. `torch.compile`)

Nsight Systems trace of the two PyTorch baselines from `experiments/benchmark_batched_gemv.py`
(`torch` = eager `A @ b[:, :, None]`, `torch_compile` = the same function under
`torch.compile(dynamic=False, fullgraph=True)`), for k = 2..99 and all 7 dtype combinations
(686 configurations). A100-SXM4-80GB, PyTorch 2.6.0a0 (NGC 24.12), cuBLAS 12.3.

## How it was produced

- Script: `experiments/profile_batched_gemv_torch.py`. It uses the same problem sizes as the
  benchmark (`n = 2**27 // k**2`, column-major A). Each configuration does 1 first call
  (which triggers compilation), 2 warmup iterations, then 3 profiled iterations. NVTX ranges
  are `first_call <alg> k=.. A=.. b=..` and `run <alg> k=.. A=.. b=..`.
- Profiling command (run from `experiments/`):
  ```
  TORCH_LOGS=output_code nsys profile --trace=cuda,nvtx,cublas --sample=none --cpuctxsw=none \
      -o ../results/nsys_batched_gemv/profile python profile_batched_gemv_torch.py
  ```
  `TORCH_LOGS=output_code` writes the code Inductor generated for every configuration to `run.log`.
- Analysis: `nsys export --type sqlite`, then `experiments/analyze_nsys_batched_gemv.py`, which
  attributes each kernel to an NVTX `run ...` range through the CUDA runtime launch call.
- nsys's own cuBLAS tracing (`--trace=cublas`) recorded no API calls. The API names below come
  from a separate run with k = 5 and 8 under `CUBLAS_LOGINFO_DBG=1` and `CUBLASLT_LOG_LEVEL=2`.

## Files

| File | Content |
|---|---|
| `profile.nsys-rep`, `profile.sqlite` | Raw trace (open the `.nsys-rep` in the Nsight Systems GUI) |
| `run.log` | stdout plus the Inductor-generated code for every compiled configuration |
| `kernels_per_config.csv` | Full demangled kernel names per configuration (raw, very long) |
| `kernels_per_config_compressed.csv` | Same data with short kernel names, see below |
| `summary_raw.txt` | `kernels_per_config_compressed.csv` grouped into k ranges with identical kernel sequences |

### `kernels_per_config_compressed.csv`

One row per (algorithm, k, A_dtype, b_dtype).

| Column | Meaning |
|---|---|
| `algorithm` | `torch` (eager) or `torch_compile` |
| `k`, `A_dtype`, `b_dtype` | Problem configuration; `n = 2**27 // k**2` |
| `kernels_per_iter` | Number of kernel launches in **one** call. Computed as (all kernels in the `run` range) / 3. |
| `gpu_time_ms_per_iter` | Sum of kernel durations in the `run` range divided by 3. This is pure GPU time: it excludes launch gaps and CPU overhead, so it is a bit lower than the CUDA-event timings in `benchmark_batched_gemv.csv`. It comes from only 3 iterations, so differences of a few percent are noise. |
| `ks` | The kernel sequence of one call, in launch order. Consecutive identical kernels are collapsed: `name xN` means N back-to-back launches. |

Kernel names in `ks`:

| Short name | Origin | Full name prefix |
|---|---|---|
| `cublas gemvx<T7=b,i>` | cuBLAS gemv kernel | `internal::gemvx::kernel<int,int,T..., (bool)b, ..., (int)i, ...>`. `T7` is the first bool template parameter, which selects one of two `enable_if` overloads, and `i` is the next int parameter. Their exact meaning is undocumented; they are kept only to tell variants apart. |
| `cublas gemv2N<a,b,c,d>` | cuBLAS gemv kernel | `gemv2N_kernel<int,int,T..., (int)a,(int)b,(int)c,(int)d, ...>` |
| `cublas gemvNSP` | cuBLAS gemv kernel | `gemvNSP_kernel<T..., 1,32,4,1024, ...>` |
| `cublas cutlass_wmma_tensorop_16x16` | cuBLAS tensor-core GEMM | `cutlass::Kernel2<cutlass_80_wmma_tensorop_{f16,bf16}_s161616gemm_..._16x16_32x1_nn_align8>` |
| `aten vectorized_elementwise_kernel`, `aten unrolled_elementwise_kernel` | PyTorch eager | dtype casts (`.to(A.dtype)`, `.to(b.dtype)`) |
| `triton_poi_fused_*` | Inductor-generated Triton | casts and padding, see below |

All cuBLAS gemv kernels above take `cublasGemvParamsEx<..., cublasGemvTensorStridedBatched<...>>`,
so they are cuBLAS's internal strided-batched gemv implementation.

## Findings

### 1. Which cuBLAS functions PyTorch calls

`A @ b[:, :, None]` → `aten::bmm` → cuBLAS **GEMM** strided-batched with `n = 1`:

- fp32: `cublasSgemmStridedBatched`
- fp64: `cublasDgemmStridedBatched`
- fp16/bf16: `cublasGemmStridedBatchedEx`

The arguments are `transa = N`, `transb = T`, `m = k_mat = k`, `n = 1`, `batchCount = n`.
cuBLAS forwards these calls to cublasLt (`cublasLt{SSS,DDD,HSH,TST}Matmul`).
PyTorch never calls `cublas*gemvStridedBatched` itself. Inside cuBLAS, however, the `n = 1`
GEMM is usually routed to the strided-batched gemv kernels (`gemvx`, `gemv2N`, `gemvNSP`).

### 2. Exceptions and side effects

- **fp16/bf16 with k a multiple of 8 (8, 16, ..., 72):** cuBLAS picks a real tensor-core GEMM,
  `cutlass_80_wmma_tensorop_*_16x16`, instead of a gemv kernel. With n = 1 most of each 16x16
  output tile is wasted work.
- **Batch chunking:** cuBLAS splits the batch into chunks of 65,535 matrices, one kernel launch
  per chunk. That gives about `ceil(n / 65535)` launches per call: 512 (+1 tail kernel) at k = 2,
  82 at k = 5, 8 at k = 16, 1 for k ≥ 46.
- **Which gemv kernel cuBLAS picks** depends on k and dtype and changes irregularly. For example,
  fp32 uses `gemvx` for k ≤ 16, `gemv2N` for 17–40, `gemvx` for 41–48, `gemv2N` for 49–72 and
  `gemvx` for 73–99. See `summary_raw.txt` for every combination.

### 3. When `torch.compile` changes anything

| Case | What Inductor generates | Effect (compile / eager time) |
|---|---|---|
| fp32/fp32, fp64/fp64 | `extern_kernels.bmm` only, no extra kernels | Same kernels and same launch counts. Ratio ≈ 1.00 |
| mixed dtypes, k ≥ 8 (also fp16, k = 7) | Triton cast kernels instead of ATen casts; `bmm` unchanged | Same cuBLAS kernels. Ratio ≈ 0.95–0.98 from slightly faster casts |
| **fp16/bf16 A, k ≤ 7** (except fp16, k = 7) | **Padding:** `triton_poi_fused_0` pads A to `(n, 8, 8)`, `triton_poi_fused_1` pads and casts b to `(n, 8, 1)`, `bmm` runs on the padded shapes, then `triton_poi_fused__to_copy_2` slices the result and casts it back | cuBLAS takes the k = 8 tensor-core path (`cutlass_wmma`) instead of a gemv kernel. Ratio **0.46–0.66** (0.81–0.91 for fp16 at k = 3 and 5) |

The padding comes from Inductor's matmul-padding pass. Padding was applied for bf16 at k = 7 but
not for fp16 at k = 7, which fits that pass deciding by benchmarking padded against unpadded at
compile time. I have not confirmed this in the PyTorch source. If it does benchmark, the choice
for borderline sizes may differ between runs or machines.

GPU-time ratios from this trace (`torch_compile` / `torch`, `gpu_time_ms_per_iter`):

| k | bf16/f32 | bf16/f64 | f16/f32 | f16/f64 | f32/f32 | f32/f64 | f64/f64 |
|---|---|---|---|---|---|---|---|
| 2 | 0.59 | 0.60 | 0.56 | 0.56 | 1.13* | 0.99 | 1.00 |
| 3 | 0.55 | 0.55 | 0.87 | 0.87 | 0.98 | 1.00 | 1.00 |
| 4 | 0.61 | 0.62 | 0.58 | 0.58 | 0.99 | 1.00 | 1.00 |
| 5 | 0.54 | 0.55 | 0.81 | 0.82 | 1.00 | 1.00 | 1.00 |
| 6 | 0.50 | 0.50 | 0.47 | 0.48 | 0.99 | 1.00 | 1.00 |
| 7 | 0.46 | 0.47 | 0.98 | 0.98 | 0.99 | 0.99 | 1.00 |
| 8 | 0.98 | 0.98 | 1.18* | 0.98 | 0.99 | 0.99 | 0.99 |
| 16 | 0.97 | 0.96 | 0.97 | 0.96 | 0.98 | 0.96 | 0.98 |
| 64 | 0.97 | 0.95 | 0.97 | 0.95 | 0.95 | 0.95 | 0.95 |
| 99 | 0.97 | 0.94 | 0.97 | 0.94 | 0.95 | 0.97 | 0.94 |

\* Both runs used identical kernels, so these values are noise from having only 3 iterations.
These ratios agree with `results/benchmark_batched_gemv.csv`.

### 4. Precision settings in this container

The NGC container sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, so
`torch.backends.cuda.matmul.allow_tf32 == True`. The cuBLAS log shows
`CUBLAS_TF32_TENSOR_OP_MATH` / `COMPUTE_32F_FAST_TF32` for fp32. This had no effect here,
because every fp32 configuration ran SIMT gemv kernels and no tensor-core kernel. It would matter
if cuBLAS ever chose a tensor-core GEMM for fp32. Reduced-precision reduction is also enabled for
fp16/bf16 (`allow_{fp16,bf16}_reduced_precision_reduction == True`).
