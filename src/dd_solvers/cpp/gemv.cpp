#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include "gemv_kernels.hpp"

namespace dd_solvers_gemv {

at::Tensor gemvStridedBatched(
    const at::Tensor& mat,  // shape: (n, k, k) row-major
    const at::Tensor& vec   // shape: (n, k)
) {
  TORCH_CHECK(
      mat.dim() == 3 && vec.dim() == 2 && mat.size(0) == vec.size(0) &&
          mat.size(1) == mat.size(2) && mat.size(1) == vec.size(1),
      "Input tensors must have compatible dimensions: (n, k, k) and (n, k)");
  TORCH_CHECK(mat.stride(1) == 1 && mat.stride(2) == mat.size(1),
              "Matrices must be in column-major format");

  int n = mat.size(0);
  int k = mat.size(1);

  at::Tensor out = at::empty_like(vec);

  if (mat.scalar_type() == at::ScalarType::Double &&
      vec.scalar_type() == at::ScalarType::Double) {
    gemvStridedBatchedDoubleDouble(mat.const_data_ptr<double>(),
                                   vec.const_data_ptr<double>(),
                                   out.mutable_data_ptr<double>(), n, k);
  } else if (mat.scalar_type() == at::ScalarType::Float &&
             vec.scalar_type() == at::ScalarType::Float) {
    gemvStridedBatchedFloatFloat(mat.const_data_ptr<float>(),
                                 vec.const_data_ptr<float>(),
                                 out.mutable_data_ptr<float>(), n, k);
  } else if (mat.scalar_type() == at::ScalarType::BFloat16 &&
             vec.scalar_type() == at::ScalarType::Float) {
    gemvStridedBatchedFloatBf16(mat.const_data_ptr<at::BFloat16>(),
                                vec.const_data_ptr<float>(),
                                out.mutable_data_ptr<float>(), n, k);
  } else if (mat.scalar_type() == at::ScalarType::Half &&
             vec.scalar_type() == at::ScalarType::Float) {
    gemvStridedBatchedFloatHalf(mat.const_data_ptr<at::Half>(),
                                vec.const_data_ptr<float>(),
                                out.mutable_data_ptr<float>(), n, k);
  } else {
    TORCH_CHECK(false, "Unsupported data type for matvec operation");
  }

  return out;
}

TORCH_LIBRARY(dd_solvers_gemv, m) {
  m.def("gemv_strided_batched", &gemvStridedBatched);
}

}  // namespace dd_solvers_gemv
