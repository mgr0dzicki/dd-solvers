#ifndef GEMV_KERNELS_HPP
#define GEMV_KERNELS_HPP

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

void gemvStridedBatchedDoubleDouble(const double* mat,
                                    const double* vec,
                                    double* out,
                                    int n,
                                    int k);

void gemvStridedBatchedFloatFloat(const float* mat,
                                  const float* vec,
                                  float* out,
                                  int n,
                                  int k);

void gemvStridedBatchedFloatBf16(const at::BFloat16* mat,
                                 const float* vec,
                                 float* out,
                                 int n,
                                 int k);

void gemvStridedBatchedFloatHalf(const at::Half* mat,
                                 const float* vec,
                                 float* out,
                                 int n,
                                 int k);

void gemvStridedBatchedDoubleBf16(const at::BFloat16* mat,
                                  const double* vec,
                                  double* out,
                                  int n,
                                  int k);

void gemvStridedBatchedDoubleHalf(const at::Half* mat,
                                  const double* vec,
                                  double* out,
                                  int n,
                                  int k);

#endif  // GEMV_KERNELS_HPP
