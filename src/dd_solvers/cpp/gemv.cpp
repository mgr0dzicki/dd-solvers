#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include "gemv_kernels.hpp"

namespace dd_solvers_gemv {

at::Tensor gemvStridedBatched(
    const at::Tensor& mat,  // shape: (n, k, k) row-major
    const at::Tensor& vec,  // shape: (n, k)
    bool use_shared_memory) {
  TORCH_CHECK(
      mat.dim() == 3 && vec.dim() == 2 && mat.size(0) == vec.size(0) &&
          mat.size(1) == mat.size(2) && mat.size(1) == vec.size(1),
      "Input tensors must have compatible dimensions: (n, k, k) and (n, k)");
  TORCH_CHECK(mat.stride(1) == 1 && mat.stride(2) == mat.size(1),
              "Matrices must be in column-major format");

  int n = mat.size(0);
  int k = mat.size(1);

  at::Tensor out = at::empty_like(vec);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, mat.scalar_type(),
      "gemv_mat_dispatch", ([&] {
        using MatType = scalar_t;

        AT_DISPATCH_FLOATING_TYPES(
            vec.scalar_type(), "gemv_vec_dispatch", ([&] {
              using VecType = scalar_t;

              if constexpr (std::is_same<MatType, double>::value &&
                            std::is_same<VecType, float>::value) {
                TORCH_CHECK(false,
                            "Double-Float mixed precision is not supported.");
              } else {
                if (use_shared_memory) {
                  gemvStridedBatchedSharedLaunch(mat.const_data_ptr<MatType>(),
                                                 vec.const_data_ptr<VecType>(),
                                                 out.data_ptr<VecType>(), n, k);
                } else {
                  gemvStridedBatchedLaunch(mat.const_data_ptr<MatType>(),
                                           vec.const_data_ptr<VecType>(),
                                           out.data_ptr<VecType>(), n, k);
                }
              }
            }));
      }));

  return out;
}

TORCH_LIBRARY(dd_solvers_gemv, m) {
  m.def("gemv_strided_batched", &gemvStridedBatched);
}

}  // namespace dd_solvers_gemv
