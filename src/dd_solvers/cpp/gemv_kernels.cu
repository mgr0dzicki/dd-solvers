#include "gemv_kernels.hpp"

__global__ void gemvStridedBatchedFloatBf16Kernel(
    const __nv_bfloat16* __restrict__ mat,  // shape: (n, k, k)
                                            // column-major
    const float* __restrict__ vec,          // shape: (n, k)
    float* __restrict__ out,                // shape: (n, k)
    int n,
    int k) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n * k)
    return;

  const int batch = tid / k;
  const int row = tid % k;

  const __nv_bfloat16* matBase = mat + (size_t)batch * (size_t)k * (size_t)k;
  const float* vecBase = vec + (size_t)batch * (size_t)k;
  float acc = 0.0f;

  for (int c = 0; c < k; c++) {
    __nv_bfloat16 m = (matBase + (size_t)c * (size_t)k)[row];
    float mf = __bfloat162float(m);
    float vf = vecBase[c];
    acc += mf * vf;
  }

  out[(size_t)batch * (size_t)k + (size_t)row] = acc;
}

__global__ void gemvStridedBatchedFloatHalfKernel(
    const __half* __restrict__ mat,  // shape: (n, k, k) column-major
    const float* __restrict__ vec,   // shape: (n, k)
    float* __restrict__ out,         // shape: (n, k)
    int n,
    int k) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n * k)
    return;

  const int batch = tid / k;
  const int row = tid % k;

  const __half* matBase = mat + (size_t)batch * (size_t)k * (size_t)k;
  const float* vecBase = vec + (size_t)batch * (size_t)k;
  float acc = 0.0f;

  for (int c = 0; c < k; c++) {
    __half m = (matBase + (size_t)c * (size_t)k)[row];
    float mf = __half2float(m);
    float vf = vecBase[c];
    acc += mf * vf;
  }

  out[(size_t)batch * (size_t)k + (size_t)row] = acc;
}

template <typename T>
__global__ void gemvStridedBatchedUniformKernel(
    const T* __restrict__ mat,  // shape: (n, k, k) column-major
    const T* __restrict__ vec,  // shape: (n, k)
    T* __restrict__ out,        // shape: (n, k)
    int n,
    int k) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n * k)
    return;

  const int batch = tid / k;
  const int row = tid % k;

  const T* matBase = mat + (size_t)batch * (size_t)k * (size_t)k;
  const T* vecBase = vec + (size_t)batch * (size_t)k;
  T acc = 0;

  for (int c = 0; c < k; c++) {
    T m = (matBase + (size_t)c * (size_t)k)[row];
    T v = vecBase[c];
    acc += m * v;
  }

  out[(size_t)batch * (size_t)k + (size_t)row] = acc;
}

void gemvStridedBatchedDoubleDouble(const double* mat,
                                    const double* vec,
                                    double* out,
                                    int n,
                                    int k) {
  const int total = n * k;
  const int blockSize = 128;
  const int grid = (total + blockSize - 1) / blockSize;

  auto stream = at::cuda::getCurrentCUDAStream();
  gemvStridedBatchedUniformKernel<double>
      <<<grid, blockSize, 0, stream>>>(mat, vec, out, n, k);
}

void gemvStridedBatchedFloatFloat(const float* mat,
                                  const float* vec,
                                  float* out,
                                  int n,
                                  int k) {
  const int total = n * k;
  const int blockSize = 128;
  const int grid = (total + blockSize - 1) / blockSize;

  auto stream = at::cuda::getCurrentCUDAStream();
  gemvStridedBatchedUniformKernel<float>
      <<<grid, blockSize, 0, stream>>>(mat, vec, out, n, k);
}

void gemvStridedBatchedFloatBf16(const at::BFloat16* mat,
                                 const float* vec,
                                 float* out,
                                 int n,
                                 int k) {
  const int total = n * k;
  const int blockSize = 128;
  const int grid = (total + blockSize - 1) / blockSize;

  auto stream = at::cuda::getCurrentCUDAStream();
  gemvStridedBatchedFloatBf16Kernel<<<grid, blockSize, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(mat), vec, out, n, k);
}

void gemvStridedBatchedFloatHalf(const at::Half* mat,
                                 const float* vec,
                                 float* out,
                                 int n,
                                 int k) {
  const int total = n * k;
  const int blockSize = 128;
  const int grid = (total + blockSize - 1) / blockSize;

  auto stream = at::cuda::getCurrentCUDAStream();
  gemvStridedBatchedFloatHalfKernel<<<grid, blockSize, 0, stream>>>(
      reinterpret_cast<const __half*>(mat), vec, out, n, k);
}
