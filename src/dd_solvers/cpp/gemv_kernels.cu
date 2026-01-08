#include "gemv_kernels.hpp"

template <typename T>
__device__ __forceinline__ T upcastLowPrecision(T v) {
  return v;
}

__device__ __forceinline__ float upcastLowPrecision(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ float upcastLowPrecision(__half v) {
  return __half2float(v);
}

template <typename T>
struct ToCudaType {
  using t = T;
};

template <>
struct ToCudaType<at::BFloat16> {
  using t = __nv_bfloat16;
};

template <>
struct ToCudaType<at::Half> {
  using t = __half;
};

__global__ void gemvStridedBatchedFloatHalfKernelShared(
    const __half* __restrict__ mat,  // shape: (n, k, k) column-major
    const float* __restrict__ vec,   // shape: (n, k)
    float* __restrict__ out,         // shape: (n, k)
    int n,
    int k) {
  const int blockSize = blockDim.x;
  const int batchPerBlock = blockSize / k;

  const int localBatch = threadIdx.x / k;
  if (localBatch >= batchPerBlock)
    return;
  const int batch = blockIdx.x * batchPerBlock + localBatch;
  const int row = threadIdx.x % k;
  if (batch >= n)
    return;

  const __half* matBase = mat + (size_t)batch * (size_t)k * (size_t)k;
  const float* vecBase = vec + (size_t)batch * (size_t)k;
  float acc = 0.0f;

  extern __shared__ float vecSharedF[];
  vecSharedF[threadIdx.x] = vecBase[row];
  __syncthreads();

  for (int c = 0; c < k; c++) {
    __half m = (matBase + (size_t)c * (size_t)k)[row];
    float mf = __half2float(m);
    float vf = vecSharedF[localBatch * k + c];
    acc += mf * vf;
  }

  out[(size_t)batch * (size_t)k + (size_t)row] = acc;
}

template <typename T, typename U>
__global__ void gemvStridedBatchedKernel(
    const T* __restrict__ mat,  // shape: (n, k, k) column-major
    const U* __restrict__ vec,  // shape: (n, k)
    U* __restrict__ out,        // shape: (n, k)
    int n,
    int k) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n * k)
    return;

  const int batch = tid / k;
  const int row = tid % k;

  const T* matBase = mat + (size_t)batch * (size_t)k * (size_t)k;
  const U* vecBase = vec + (size_t)batch * (size_t)k;
  U acc = 0;

  for (int c = 0; c < k; c++) {
    T m = (matBase + (size_t)c * (size_t)k)[row];
    U mv = static_cast<U>(upcastLowPrecision(m));
    U v = vecBase[c];
    acc += mv * v;
  }

  out[(size_t)batch * (size_t)k + (size_t)row] = acc;
}

template <typename T, typename U>
void gemvStridedBatchedLaunch(const T* mat,
                              const U* vec,
                              U* out,
                              int n,
                              int k) {
  const int total = n * k;
  const int blockSize = 128;
  const int grid = (total + blockSize - 1) / blockSize;

  using MatCudaType = typename ToCudaType<T>::t;
  using VecCudaType = typename ToCudaType<U>::t;

  auto stream = at::cuda::getCurrentCUDAStream();
  gemvStridedBatchedKernel<<<grid, blockSize, 0, stream>>>(
      reinterpret_cast<const MatCudaType*>(mat),
      reinterpret_cast<const VecCudaType*>(vec),
      reinterpret_cast<VecCudaType*>(out), n, k);
}

template void gemvStridedBatchedLaunch<at::BFloat16, float>(const at::BFloat16*,
                                                            const float*,
                                                            float*,
                                                            int,
                                                            int);

template void gemvStridedBatchedLaunch<at::BFloat16, double>(
    const at::BFloat16*,
    const double*,
    double*,
    int,
    int);

template void gemvStridedBatchedLaunch<at::Half, float>(const at::Half*,
                                                        const float*,
                                                        float*,
                                                        int,
                                                        int);

template void gemvStridedBatchedLaunch<at::Half, double>(const at::Half*,
                                                         const double*,
                                                         double*,
                                                         int,
                                                         int);

template void gemvStridedBatchedLaunch<float, float>(const float*,
                                                     const float*,
                                                     float*,
                                                     int,
                                                     int);

template void gemvStridedBatchedLaunch<float, double>(const float*,
                                                      const double*,
                                                      double*,
                                                      int,
                                                      int);

template void gemvStridedBatchedLaunch<double, double>(const double*,
                                                       const double*,
                                                       double*,
                                                       int,
                                                       int);

void gemvStridedBatchedFloatHalf(const at::Half* mat,
                                 const float* vec,
                                 float* out,
                                 int n,
                                 int k) {
  if (k <= 8) {
    const int total = n * k;
    const int blockSize = 128;
    const int grid = (total + blockSize - 1) / blockSize;

    auto stream = at::cuda::getCurrentCUDAStream();
    gemvStridedBatchedKernel<__half, float><<<grid, blockSize, 0, stream>>>(
        reinterpret_cast<const __half*>(mat), vec, out, n, k);
  } else {
    int batchPerBlock = 256 / k;
    int blockSize = batchPerBlock * k;
    if (k > 128) {
      blockSize = (k + 31) / 32 * 32;
      batchPerBlock = 1;
    }

    const int grid = (n + batchPerBlock - 1) / batchPerBlock;

    auto stream = at::cuda::getCurrentCUDAStream();
    gemvStridedBatchedFloatHalfKernelShared<<<
        grid, blockSize, sizeof(float) * batchPerBlock * k, stream>>>(
        reinterpret_cast<const __half*>(mat), vec, out, n, k);
  }
}

void gemvStridedBatchedDoubleHalf(const at::Half* mat,
                                  const double* vec,
                                  double* out,
                                  int n,
                                  int k) {
  const int total = n * k;

  int batchPerBlock = 256 / k;
  int blockSize = batchPerBlock * k;
  if (k > 128) {
    blockSize = (k + 31) / 32 * 32;
    batchPerBlock = 1;
  }

  const int grid = (n + batchPerBlock - 1) / batchPerBlock;

  auto stream = at::cuda::getCurrentCUDAStream();
  gemvStridedBatchedKernel<__half, double>
      <<<grid, blockSize, sizeof(double) * batchPerBlock * k, stream>>>(
          reinterpret_cast<const __half*>(mat), vec, out, n, k);
}
