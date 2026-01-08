#ifndef GEMV_KERNELS_HPP
#define GEMV_KERNELS_HPP

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

void gemvStridedBatchedLaunch(const double* mat,
                              const double* vec,
                              double* out,
                              int n,
                              int k);

void gemvStridedBatchedLaunch(const float* mat,
                              const double* vec,
                              double* out,
                              int n,
                              int k);

void gemvStridedBatchedLaunch(const float* mat,
                              const float* vec,
                              float* out,
                              int n,
                              int k);

void gemvStridedBatchedLaunch(const at::BFloat16* mat,
                              const float* vec,
                              float* out,
                              int n,
                              int k);

void gemvStridedBatchedLaunch(const at::Half* mat,
                              const float* vec,
                              float* out,
                              int n,
                              int k);

void gemvStridedBatchedLaunch(const at::BFloat16* mat,
                              const double* vec,
                              double* out,
                              int n,
                              int k);

void gemvStridedBatchedLaunch(const at::Half* mat,
                              const double* vec,
                              double* out,
                              int n,
                              int k);

#endif  // GEMV_KERNELS_HPP
