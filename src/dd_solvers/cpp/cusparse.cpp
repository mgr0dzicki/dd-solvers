#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include "utils.hpp"

namespace dd_solvers_cusparse {

cusparseHandle_t cusparseHandle;

void initHandles() {
  CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
}

void destroyHandles() {
  if (cusparseHandle != nullptr) {
    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
    cusparseHandle = nullptr;
  }
}

at::Tensor csrToBsr(const at::Tensor& csrMatrix, int64_t blockSize) {
  cusparseMatDescr_t descrA, descrC;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));

  int m = csrMatrix.size(0);
  int n = csrMatrix.size(1);
  int mb = (m + blockSize - 1) / blockSize;
  cusparseDirection_t blockDir = CUSPARSE_DIRECTION_COLUMN;

  at::Tensor values = csrMatrix.values();
  at::Tensor rowStart = csrMatrix.crow_indices().to(at::kInt);
  at::Tensor colIndices = csrMatrix.col_indices().to(at::kInt);

  int bufferSize;
  if (values.dtype() == at::kFloat) {
    CHECK_CUSPARSE(cusparseScsr2gebsr_bufferSize(
        cusparseHandle, blockDir, m, n, descrA,
        values.data_ptr<float>(), rowStart.data_ptr<int>(),
        colIndices.data_ptr<int>(), blockSize, blockSize, &bufferSize));
  } else if (values.dtype() == at::kDouble) {
    CHECK_CUSPARSE(cusparseDcsr2gebsr_bufferSize(
        cusparseHandle, blockDir, m, n, descrA,
        values.data_ptr<double>(), rowStart.data_ptr<int>(),
        colIndices.data_ptr<int>(), blockSize, blockSize, &bufferSize));
  } else {
    throw std::invalid_argument(
        "Unsupported values data type. Only float and double are supported.");
  }

  void* buffer;
  CHECK_CUDA(cudaMalloc(&buffer, bufferSize));

  at::Tensor bsrRowPtr = at::empty(
      {mb + 1}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  int* bsrRowPtrC = bsrRowPtr.mutable_data_ptr<int>();

  int nnzb;
  int* nnzTotalDevHostPtr = &nnzb;
  CHECK_CUSPARSE(cusparseXcsr2gebsrNnz(
      cusparseHandle, blockDir, m, n, descrA,
      rowStart.data_ptr<int>(), colIndices.data_ptr<int>(), descrC, bsrRowPtrC,
      blockSize, blockSize, nnzTotalDevHostPtr, buffer));

  if (nnzTotalDevHostPtr != nullptr) {
    nnzb = *nnzTotalDevHostPtr;
  } else {
    CHECK_CUDA(cudaMemcpy(&nnzb, bsrRowPtrC + mb, sizeof(int),
                          cudaMemcpyDeviceToHost));
    int base;
    CHECK_CUDA(
        cudaMemcpy(&base, bsrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost));
    nnzb -= base;
  }

  at::Tensor bsrColInd =
      at::empty({nnzb}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  int* bsrColIndC = bsrColInd.mutable_data_ptr<int>();

  at::Tensor bsrValues =
      at::empty_strided({nnzb, blockSize, blockSize}, {blockSize * blockSize, 1, blockSize}, // column-major
                at::TensorOptions().dtype(values.dtype()).device(at::kCUDA));
  void* bsrValC = bsrValues.mutable_data_ptr();

  if (values.dtype() == at::kFloat) {
    CHECK_CUSPARSE(cusparseScsr2gebsr(
        cusparseHandle, blockDir, m, n, descrA,
        values.data_ptr<float>(), rowStart.data_ptr<int>(),
        colIndices.data_ptr<int>(), descrC, reinterpret_cast<float*>(bsrValC),
        bsrRowPtrC, bsrColIndC, blockSize, blockSize, buffer));
  } else {
    CHECK_CUSPARSE(cusparseDcsr2gebsr(
        cusparseHandle, blockDir, m, n, descrA,
        values.data_ptr<double>(), rowStart.data_ptr<int>(),
        colIndices.data_ptr<int>(), descrC, reinterpret_cast<double*>(bsrValC),
        bsrRowPtrC, bsrColIndC, blockSize, blockSize, buffer));
  }

  CHECK_CUDA(cudaFree(buffer));
  CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
  CHECK_CUSPARSE(cusparseDestroyMatDescr(descrC));

  return at::native::sparse_bsr_tensor(bsrRowPtr, bsrColInd, bsrValues, {m, n},
                                       values.scalar_type(), at::kSparseBsr,
                                       at::kCUDA);
}

at::Tensor sortCsr(at::Tensor& csrMatrix) {
  // Note that it invalidates original tensor!
  const int m = csrMatrix.size(0);
  const int n = csrMatrix.size(1);
  const int nnz = csrMatrix.values().size(0);

  size_t pBufferSizeInBytes = 0;
  void* pBuffer = nullptr;
  int* P = nullptr;

  auto [valuesCudaType, valuesPtr] = valuesToCudaDataType(csrMatrix.values());
  at::Tensor rowStart = csrMatrix.crow_indices().to(at::kInt);
  at::Tensor colIndices = csrMatrix.col_indices().to(at::kInt);

  cusparseMatDescr_t descrA;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));

  CHECK_CUSPARSE(cusparseXcsrsort_bufferSizeExt(
      cusparseHandle, m, n, nnz, rowStart.data_ptr<int>(),
      colIndices.data_ptr<int>(), &pBufferSizeInBytes));
  CHECK_CUDA(cudaMalloc(&pBuffer, pBufferSizeInBytes));

  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&P), sizeof(int) * nnz));
  CHECK_CUSPARSE(cusparseCreateIdentityPermutation(cusparseHandle, nnz, P));

  CHECK_CUSPARSE(cusparseXcsrsort(
      cusparseHandle, m, n, nnz, descrA, rowStart.data_ptr<int>(),
      colIndices.mutable_data_ptr<int>(), P, pBuffer));

  CHECK_CUDA(cudaFree(pBuffer));

  at::Tensor valuesSorted = at::empty_like(csrMatrix.values());

  cusparseSpVecDescr_t vecP;
  cusparseDnVecDescr_t vecValues;
  CHECK_CUSPARSE(cusparseCreateSpVec(
      &vecP, nnz, nnz, P, valuesSorted.mutable_data_ptr(), CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, valuesCudaType));
  CHECK_CUSPARSE(
      cusparseCreateDnVec(&vecValues, nnz, valuesPtr, valuesCudaType));
  CHECK_CUSPARSE(cusparseGather(cusparseHandle, vecValues, vecP));
  CHECK_CUSPARSE(cusparseDestroySpVec(vecP));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecValues));

  CHECK_CUDA(cudaFree(P));
  CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));

  return at::native::sparse_csr_tensor(rowStart, colIndices, valuesSorted,
                                       {m, n}, valuesSorted.scalar_type(),
                                       at::kSparseCsr, at::kCUDA);
}

TORCH_LIBRARY(dd_solvers_cusparse, m) {
  m.def("init", &initHandles);
  m.def("destroy", &destroyHandles);
  m.def("csr_to_bsr", &csrToBsr);
  m.def("sort_csr", &sortCsr);
}

}  // namespace dd_solvers_cusparse
