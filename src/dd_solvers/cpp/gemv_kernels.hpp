#ifndef GEMV_KERNELS_HPP
#define GEMV_KERNELS_HPP

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

void gemvStridedBatchedDouble(const double* mat,
                              const double* vec,
                              double* out,
                              int n,
                              int k);

void gemvStridedBatchedFloat(const float* mat,
                             const float* vec,
                             float* out,
                             int n,
                             int k);

void gemvStridedBatchedBf16(const at::BFloat16* mat,
                            const float* vec,
                            float* out,
                            int n,
                            int k);

void gemvStridedBatchedHalf(const at::Half* mat,
                            const float* vec,
                            float* out,
                            int n,
                            int k);

#endif  // GEMV_KERNELS_HPP
