#ifndef UTILS_HPP
#define UTILS_HPP

#include <cudss.h>
#include <cusparse.h>
#include <sstream>

#define CHECK_CUDA(call)                                              \
  {                                                                   \
    cudaError_t err;                                                  \
    if ((err = (call)) != cudaSuccess) {                              \
      std::stringstream errorMessageStream;                           \
      errorMessageStream << "CUDA error: " << cudaGetErrorString(err) \
                         << " at " << __FILE__ << ":" << __LINE__;    \
      throw std::runtime_error(errorMessageStream.str());             \
    }                                                                 \
  }

#define CHECK_CUSPARSE(call)                                                   \
  {                                                                            \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS) {                           \
      std::stringstream errorMessageStream;                                    \
      errorMessageStream << "CUSPARSE error: " << cusparseGetErrorString(err)  \
                         << " at " << __FILE__ << ":" << __LINE__;             \
      cudaError_t cuda_err = cudaGetLastError();                               \
      if (cuda_err != cudaSuccess) {                                           \
        errorMessageStream << ", CUDA error: " << cudaGetErrorString(cuda_err) \
                           << " also detected";                                \
      }                                                                        \
      throw std::runtime_error(errorMessageStream.str());                      \
    }                                                                          \
  }

#define CHECK_CUDSS(call)                                                      \
  {                                                                            \
    cudssStatus_t err;                                                         \
    if ((err = (call)) != CUDSS_STATUS_SUCCESS) {                              \
      std::stringstream errorMessageStream;                                    \
      errorMessageStream << "CUDSS error: " << err << " at " << __FILE__       \
                         << ":" << __LINE__;                                   \
      cudaError_t cuda_err = cudaGetLastError();                               \
      if (cuda_err != cudaSuccess) {                                           \
        errorMessageStream << ", CUDA error: " << cudaGetErrorString(cuda_err) \
                           << " also detected";                                \
      }                                                                        \
      throw std::runtime_error(errorMessageStream.str());                      \
    }                                                                          \
  }

#define CHECK_DEVICE_TYPE(tensor, deviceType)                                  \
  {                                                                            \
    if ((tensor).device().type() != deviceType) {                              \
      std::stringstream errorMessageStream;                                    \
      errorMessageStream << "Expected tensor on device " << deviceType         \
                         << ", but got tensor on device " << (tensor).device() \
                         << " at " << __FILE__ << ":" << __LINE__;             \
      throw std::runtime_error(errorMessageStream.str());                      \
    }                                                                          \
  }

constexpr size_t sizeofCudaDataType(cudaDataType_t t) {
  switch (t) {
    case CUDA_R_32F:
      return 4;
    case CUDA_R_64F:
      return 8;
    case CUDA_R_32I:
      return 4;
    default:
      throw std::runtime_error("unsupported data type");
  }
}

inline std::pair<cudaDataType_t, void*> valuesToCudaDataType(
    const torch::Tensor& tensor) {
  if (tensor.dtype() == at::kFloat)
    return {CUDA_R_32F, tensor.mutable_data_ptr<float>()};
  if (tensor.dtype() == at::kDouble)
    return {CUDA_R_64F, tensor.mutable_data_ptr<double>()};
  throw std::runtime_error("unsupported values data type");
}

#endif  // UTILS_HPP
