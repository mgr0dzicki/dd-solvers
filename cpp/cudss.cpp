#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

#include "utils.hpp"

namespace dd_solvers_cudss {

cudssHandle_t cudssHandle;

void initHandles() {
  CHECK_CUDSS(cudssCreate(&cudssHandle));
  CHECK_CUDSS(cudssSetThreadingLayer(cudssHandle, "libcudss_mtlayer_gomp.so"));
}

void destroyHandles() {
  if (cudssHandle != nullptr) {
    CHECK_CUDSS(cudssDestroy(cudssHandle));
    cudssHandle = nullptr;
  }
}

cudssMatrix_t vectorToCudss(const at::Tensor& vec) {
  auto [valueType, values] = valuesToCudaDataType(vec);

  cudssMatrix_t cudssMatrix;
  CHECK_CUDSS(cudssMatrixCreateDn(&cudssMatrix, vec.size(0), 1, vec.size(0),
                                  values, valueType, CUDSS_LAYOUT_COL_MAJOR));

  return cudssMatrix;
}

struct CudssSolver : torch::CustomClassHolder {
  const cudssConfig_t config;
  const cudssData_t data;
  const cudssMatrix_t matrix;

  at::Tensor rowStart, colIndices, values;

  CudssSolver(const cudssConfig_t& config,
              const cudssData_t& data,
              const cudssMatrix_t& matrix,
              const at::Tensor& rowStart,
              const at::Tensor& colIndices,
              const at::Tensor& values)
      : config(config),
        data(data),
        matrix(matrix),
        rowStart(rowStart),
        colIndices(colIndices),
        values(values) {}
  ~CudssSolver();
  at::Tensor solve(const at::Tensor& rhs) const;
};

CudssSolver::~CudssSolver() {
  // No checks here, as destructor should not throw exceptions.
  cudssMatrixDestroy(matrix);
  cudssDataDestroy(cudssHandle, data);
  cudssConfigDestroy(config);
}

at::Tensor CudssSolver::solve(const at::Tensor& rhs) const {
  cudssMatrix_t rhs_cudss = vectorToCudss(rhs);

  at::Tensor x = at::empty(
      {rhs.size(0)}, at::TensorOptions().dtype(rhs.dtype()).device(at::kCUDA));
  cudssMatrix_t x_cudss = vectorToCudss(x);

  CHECK_CUDSS(cudssExecute(cudssHandle, CUDSS_PHASE_SOLVE, config, data, matrix,
                           x_cudss, rhs_cudss));

  CHECK_CUDSS(cudssMatrixDestroy(rhs_cudss));
  CHECK_CUDSS(cudssMatrixDestroy(x_cudss));

  return x;
}

c10::intrusive_ptr<CudssSolver> spdFactorize(const at::Tensor& A) {
  cudssConfig_t config;
  cudssData_t data;

  CHECK_CUDSS(cudssConfigCreate(&config));
  CHECK_CUDSS(cudssDataCreate(cudssHandle, &data));

  auto [valueType, values] = valuesToCudaDataType(A.values());
  at::Tensor rowStart = A.crow_indices().to(at::kInt);
  at::Tensor colIndices = A.col_indices().to(at::kInt);

  cudssMatrix_t cudssMatrix;
  CHECK_CUDSS(cudssMatrixCreateCsr(
      &cudssMatrix, A.size(0), A.size(1), A.values().size(0),
      rowStart.mutable_data_ptr<int>(), nullptr,
      colIndices.mutable_data_ptr<int>(), values, CUDA_R_32I, valueType,
      CUDSS_MTYPE_SPD, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));

  CHECK_CUDSS(cudssExecute(cudssHandle, CUDSS_PHASE_ANALYSIS, config, data,
                           cudssMatrix, nullptr, nullptr));
  CHECK_CUDSS(cudssExecute(cudssHandle, CUDSS_PHASE_FACTORIZATION, config, data,
                           cudssMatrix, nullptr, nullptr));

  return c10::make_intrusive<CudssSolver>(config, data, cudssMatrix, rowStart,
                                          colIndices, A.values());
}

struct CudssBatchSolver : torch::CustomClassHolder {
  const cudssConfig_t config;
  const cudssData_t data;
  const cudssMatrix_t matrixBatch;
  at::Tensor nrows, ones, nnz;  // CPU
  at::Tensor rowStart, colIndices, rowStartPtrs, colIndicesPtrs, values,
      nrowsCumSumEx;  // GPU

  CudssBatchSolver(const cudssConfig_t& config,
                   const cudssData_t& data,
                   const cudssMatrix_t& matrixBatch,
                   const at::Tensor& nrows,
                   const at::Tensor& ones,
                   const at::Tensor& nnz,
                   const at::Tensor& rowStart,
                   const at::Tensor& colIndices,
                   const at::Tensor& rowStartPtrs,
                   const at::Tensor& colIndicesPtrs,
                   const at::Tensor& values,
                   const at::Tensor& nrowsCumSumEx)
      : config(config),
        data(data),
        matrixBatch(matrixBatch),
        nrows(nrows),
        ones(ones),
        nnz(nnz),
        rowStart(rowStart),
        colIndices(colIndices),
        rowStartPtrs(rowStartPtrs),
        colIndicesPtrs(colIndicesPtrs),
        values(values),
        nrowsCumSumEx(nrowsCumSumEx) {}
  ~CudssBatchSolver();
  at::Tensor solve(const at::Tensor& rhs) const;
};

CudssBatchSolver::~CudssBatchSolver() {
  // No checks here, as destructor should not throw exceptions.
  cudssMatrixDestroy(matrixBatch);
  cudssDataDestroy(cudssHandle, data);
  cudssConfigDestroy(config);
}

at::Tensor CudssBatchSolver::solve(const at::Tensor& rhs) const {
  auto [rhsType, rhsPtr] = valuesToCudaDataType(rhs);

  cudssMatrix_t rhsCudss, xCudss;

  at::Tensor rhsValues = nrowsCumSumEx * sizeofCudaDataType(rhsType) +
                         reinterpret_cast<int64_t>(rhsPtr);
  CHECK_CUDSS(cudssMatrixCreateBatchDn(
      &rhsCudss, nrows.size(0), nrows.mutable_data_ptr<int>(),
      ones.mutable_data_ptr<int>(), nrows.mutable_data_ptr<int>(),
      reinterpret_cast<void**>(rhsValues.mutable_data_ptr<int64_t>()),
      CUDA_R_32I, rhsType, CUDSS_LAYOUT_COL_MAJOR));

  at::Tensor x = at::empty(
      {rhs.size(0)}, at::TensorOptions().dtype(rhs.dtype()).device(at::kCUDA));
  at::Tensor xValues = nrowsCumSumEx * sizeofCudaDataType(rhsType) +
                       reinterpret_cast<int64_t>(x.mutable_data_ptr());
  CHECK_CUDSS(cudssMatrixCreateBatchDn(
      &xCudss, nrows.size(0), nrows.mutable_data_ptr<int>(),
      ones.mutable_data_ptr<int>(), nrows.mutable_data_ptr<int>(),
      reinterpret_cast<void**>(xValues.mutable_data_ptr<int64_t>()), CUDA_R_32I,
      rhsType, CUDSS_LAYOUT_COL_MAJOR));

  CHECK_CUDSS(cudssExecute(cudssHandle, CUDSS_PHASE_SOLVE, config, data,
                           matrixBatch, xCudss, rhsCudss));

  CHECK_CUDSS(cudssMatrixDestroy(rhsCudss));
  CHECK_CUDSS(cudssMatrixDestroy(xCudss));

  return x;
}

at::Tensor cumSumEx(const at::Tensor& t) {
  at::Tensor result = at::empty(
      {t.size(0)}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  result[0] = 0;
  result.slice(0, 1) = t.to(at::kCUDA).slice(0, std::nullopt, -1);
  result.cumsum_(0);
  return result;
}

c10::intrusive_ptr<CudssBatchSolver> spdBatchFactorize(
    const at::Tensor& nrows,
    const at::Tensor& nnz,
    const at::Tensor& values,
    const at::Tensor& colIndices,
    const at::Tensor& rowStart) {
  CHECK_DEVICE_TYPE(values, at::kCUDA);
  CHECK_DEVICE_TYPE(colIndices, at::kCUDA);
  CHECK_DEVICE_TYPE(rowStart, at::kCUDA);

  at::Tensor colIndicesInt = colIndices.to(at::kInt);
  at::Tensor rowStartInt = rowStart.to(at::kInt);

  at::Tensor nrowsHost = nrows.to(at::kCPU, at::kInt);
  at::Tensor nnzHost = nnz.to(at::kCPU, at::kInt);
  at::Tensor onesHost = at::ones_like(nrowsHost, at::kCPU);

  at::Tensor nrowsCumSumEx = cumSumEx(nrows);
  at::Tensor nrowsPlusOneCumSumEx = cumSumEx(nrows + 1);
  at::Tensor nnzCumSumEx = cumSumEx(nnz);

  cudssConfig_t config;
  cudssData_t data;

  CHECK_CUDSS(cudssConfigCreate(&config));
  CHECK_CUDSS(cudssDataCreate(cudssHandle, &data));

  auto [valueType, valuesStart] = valuesToCudaDataType(values);
  at::Tensor rowStartPtrs =
      nrowsPlusOneCumSumEx * sizeofCudaDataType(CUDA_R_32I) +
      reinterpret_cast<int64_t>(rowStartInt.mutable_data_ptr());
  at::Tensor colIndicesPtrs =
      nnzCumSumEx * sizeofCudaDataType(CUDA_R_32I) +
      reinterpret_cast<int64_t>(colIndicesInt.mutable_data_ptr());
  at::Tensor valuesPtrs = nnzCumSumEx * sizeofCudaDataType(valueType) +
                          reinterpret_cast<int64_t>(valuesStart);

  cudssMatrix_t matrixBatch;
  CHECK_CUDSS(cudssMatrixCreateBatchCsr(
      &matrixBatch, nrows.size(0), nrowsHost.mutable_data_ptr<int>(),
      nrowsHost.mutable_data_ptr<int>(), nnzHost.mutable_data_ptr<int>(),
      reinterpret_cast<void**>(rowStartPtrs.mutable_data_ptr<int64_t>()),
      nullptr,
      reinterpret_cast<void**>(colIndicesPtrs.mutable_data_ptr<int64_t>()),
      reinterpret_cast<void**>(valuesPtrs.mutable_data_ptr<int64_t>()),
      CUDA_R_32I, valueType, CUDSS_MTYPE_SPD, CUDSS_MVIEW_FULL,
      CUDSS_BASE_ZERO));

  CHECK_CUDSS(cudssExecute(cudssHandle, CUDSS_PHASE_ANALYSIS, config, data,
                           matrixBatch, nullptr, nullptr));
  CHECK_CUDSS(cudssExecute(cudssHandle, CUDSS_PHASE_FACTORIZATION, config, data,
                           matrixBatch, nullptr, nullptr));

  return c10::make_intrusive<CudssBatchSolver>(
      config, data, matrixBatch, nrowsHost, onesHost, nnzHost, rowStartInt,
      colIndicesInt, rowStartPtrs, colIndicesPtrs, valuesPtrs, nrowsCumSumEx);
}

TORCH_LIBRARY(dd_solvers_cudss, m) {
  m.class_<CudssSolver>("CudssSolver").def("solve", &CudssSolver::solve);
  m.class_<CudssBatchSolver>("CudssBatchSolver")
      .def("solve", &CudssBatchSolver::solve);

  m.def("init", &initHandles);
  m.def("destroy", &destroyHandles);
  m.def("spd_factorize", &spdFactorize);
  m.def("spd_batch_factorize", &spdBatchFactorize);
}

}  // namespace dd_solvers_cudss
