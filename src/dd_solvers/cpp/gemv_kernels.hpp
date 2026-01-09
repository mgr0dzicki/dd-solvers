#ifndef GEMV_KERNELS_HPP
#define GEMV_KERNELS_HPP

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

template <typename T, typename U>
void gemvStridedBatchedLaunch(const T* mat, const U* vec, U* out, int n, int k);

template <typename T, typename U>
void gemvStridedBatchedSharedLaunch(const T* mat,
                                    const U* vec,
                                    U* out,
                                    int n,
                                    int k);

#endif  // GEMV_KERNELS_HPP
