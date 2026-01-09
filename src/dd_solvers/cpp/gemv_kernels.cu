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
__global__ void gemvStridedBatchedSharedKernel(
    const T* __restrict__ mat,  // shape: (n, k, k) column-major
    const U* __restrict__ vec,  // shape: (n, k)
    U* __restrict__ out,        // shape: (n, k)
    int n,
    int k) {
  const int batchPerBlock = blockDim.x / k;
  const int localBatch = threadIdx.x / k;
  if (localBatch >= batchPerBlock)
    return;

  const int batch = blockIdx.x * batchPerBlock + localBatch;
  const int row = threadIdx.x % k;
  if (batch >= n)
    return;

  const T* matBase = mat + (size_t)batch * (size_t)k * (size_t)k;
  const U* vecBase = vec + (size_t)batch * (size_t)k;
  U acc = 0;

  extern __shared__ unsigned char vecSharedBuff[];
  U* vecShared = reinterpret_cast<U*>(vecSharedBuff);
  vecShared[threadIdx.x] = vecBase[row];
  __syncthreads();

  for (int c = 0; c < k; c++) {
    T m = (matBase + (size_t)c * (size_t)k)[row];
    U mv = static_cast<U>(upcastLowPrecision(m));
    U v = vecShared[localBatch * k + c];
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

template <typename T, typename U>
void gemvStridedBatchedSharedLaunch(const T* mat,
                                    const U* vec,
                                    U* out,
                                    int n,
                                    int k) {
  int batchPerBlock = 256 / k;
  int blockSize = 256;
  if (k > 128) {
    blockSize = (k + 31) / 32 * 32;
    batchPerBlock = 1;
  }

  const int grid = (n + batchPerBlock - 1) / batchPerBlock;

  using MatCudaType = typename ToCudaType<T>::t;
  using VecCudaType = typename ToCudaType<U>::t;

  auto stream = at::cuda::getCurrentCUDAStream();
  gemvStridedBatchedSharedKernel<<<grid, blockSize,
                                   blockSize * sizeof(VecCudaType), stream>>>(
      reinterpret_cast<const MatCudaType*>(mat),
      reinterpret_cast<const VecCudaType*>(vec),
      reinterpret_cast<VecCudaType*>(out), n, k);
}

#define INSTANTIATE_GEMV(TMat, TVec)                                           \
  template void gemvStridedBatchedLaunch(const TMat*, const TVec*, TVec*, int, \
                                         int);                                 \
  template void gemvStridedBatchedSharedLaunch(const TMat*, const TVec*,       \
                                               TVec*, int, int);

INSTANTIATE_GEMV(at::BFloat16, float);
INSTANTIATE_GEMV(at::BFloat16, double);
INSTANTIATE_GEMV(at::Half, float);
INSTANTIATE_GEMV(at::Half, double);
INSTANTIATE_GEMV(float, float);
INSTANTIATE_GEMV(float, double);
INSTANTIATE_GEMV(double, double);
