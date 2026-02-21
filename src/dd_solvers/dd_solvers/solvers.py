from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, NamedTuple, Tuple

import numba

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

from numba import cuda
import numpy as np
import scipy.sparse as sps
import gc

import torch
import pyamgx

from .bsr import FastBSR
from .cusparse import csr_to_bsr, sort_csr, gather_csr
from .gemv import gemv_strided_batched
from . import cudss

pyamgx.initialize()


Metadata = dict[str, Any]


class ConvergenceNotAchieved(Exception):
    def __init__(self, metadata: Metadata):
        self.metadata = metadata
        if "iterations" not in self.metadata:
            msg = "Convergence not achieved."
        else:
            msg = (
                f"Convergence not achieved in {self.metadata['iterations']} iterations."
            )
        super().__init__(msg)


def collect():
    """
    Collects garbage and clears CUDA cache.
    """
    gc.collect()
    torch.cuda.empty_cache()


def timeit(fun, warmup_iters=10, iters=10, repetitions=20):
    times = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(warmup_iters):
            fun()

        start.record()
        for _ in range(iters):
            fun()
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end) / iters)

    return min(times)


class SparseSolver(ABC):
    @abstractmethod
    def setup(self, matrix: torch.Tensor, *args, **kwargs) -> Metadata | None:
        """
        Setup the solver with the given matrix in CSR format.
        """
        pass

    @abstractmethod
    def solve(self, rhs: torch.Tensor) -> Tuple[torch.Tensor, Metadata]:
        pass

    @abstractmethod
    def destroy(self) -> None:
        pass


class CUDSS(SparseSolver):
    def __str__(self):
        return "CUDSS"

    def setup(self, matrix: torch.Tensor, *args, **kwargs) -> None:
        assert matrix.is_cuda
        assert matrix.dtype in [torch.float32, torch.float64]
        assert matrix.values().shape[0] < 2**31

        matrix_32 = torch.sparse_csr_tensor(
            matrix.crow_indices().to(torch.int32),
            matrix.col_indices().to(torch.int32),
            matrix.values(),
            matrix.shape,
        )
        del matrix
        collect()

        self.solver = cudss.spd_factorize(matrix_32)
        self.dtype = matrix_32.dtype

    def solve(self, rhs: torch.Tensor) -> Tuple[torch.Tensor, Metadata]:
        assert rhs.is_cuda
        assert rhs.dtype == self.dtype

        return self.solver.solve(rhs), {}

    def destroy(self):
        self.solver = None


def csr_to_torch(csr: sps.csr_matrix) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.as_tensor(csr.indptr),
        torch.as_tensor(csr.indices),
        torch.as_tensor(csr.data),
        csr.shape,
    )


def permute_csr_tensor(mat: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """
    Returns CSR matrix `Ap`, such that `Ap[perm][:, perm] == A`.
    """
    nnz = mat.col_indices().shape[0]
    indptr = mat.crow_indices()
    indices = mat.col_indices()
    values = mat.values()
    shape = mat.shape
    del mat

    row_sizes = indptr[1:] - indptr[:-1]
    indptr2 = torch.empty_like(indptr)
    indptr2[0] = 0
    indptr2[1:][perm] = row_sizes
    indptr2.cumsum_(dim=0)

    asort = indptr2[perm].repeat_interleave(row_sizes)
    asort += torch.arange(nnz, device=asort.device)
    asort -= indptr[:-1].repeat_interleave(row_sizes)

    del row_sizes, indptr
    collect()

    # Since index_select consumes a lot of memory for large tensors,
    # we process indices and values in chunks.
    CHUNK_SIZE = 10**9

    indices2 = torch.empty_like(indices)
    for start in range(0, nnz, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, nnz)
        indices2[asort[start:end]] = perm[indices[start:end]].to(dtype=indices2.dtype)
    del indices
    collect()

    values2 = torch.empty_like(values)
    for start in range(0, nnz, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, nnz)
        values2[asort[start:end]] = values[start:end]
    del values, asort
    collect()

    return sort_csr(torch.sparse_csr_tensor(indptr2, indices2, values2, shape))


def _fused_add(x, y, alpha):
    x += alpha * y


fused_add = torch.compile(_fused_add)


def _fused_sub(x, y, alpha):
    x -= alpha * y


fused_sub = torch.compile(_fused_sub)


def _fused_mul_add(p, z, beta):
    p *= beta
    p += z


fused_mul_add = torch.compile(_fused_mul_add)


class CG(SparseSolver):
    def __init__(
        self,
        preconditioner: SparseSolver | None = None,
        rtol: float = 1e-7,
        atol: float = 0,
        maxiter: int | None = 1000,
        bsr_matmul: bool = True,
        estimate_cond: bool = False,
    ):
        self.preconditioner = preconditioner
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.bsr_matmul = bsr_matmul
        self.estimate_cond = estimate_cond

    def __str__(self):
        return f"CG({self.preconditioner or ''}, bsr_matmul={self.bsr_matmul})"

    def setup(self, matrix: torch.Tensor, *args, **kwargs) -> None:
        if self.preconditioner is not None:
            self.preconditioner.setup(matrix, *args, **kwargs)

        if self.bsr_matmul:
            if "block_size" in kwargs:
                block_size = kwargs["block_size"]
            else:
                raise ValueError(
                    "Unable to determine block size for BSR matrix multiplication."
                )

            self.matrix = FastBSR(csr_to_bsr(matrix, block_size=block_size))
        else:
            self.matrix = matrix

    def solve(self, rhs: torch.Tensor, x0=None) -> Tuple[torch.Tensor, Metadata]:
        atol = max(self.atol, self.rtol * torch.linalg.norm(rhs).item())
        n = len(rhs)
        maxiter = self.maxiter or n * 10
        x = x0 if x0 is not None else torch.zeros_like(rhs)
        if self.preconditioner is not None and hasattr(self.preconditioner, "T0"):
            x += self.preconditioner.T0(rhs)
        r = rhs - self.matrix @ x
        rho_prev, p = None, None

        residual_norms = []
        preconditioner_metadata = []
        if self.estimate_cond:
            alphas = []
            betas = []
        for i in range(maxiter):
            res_norm = torch.linalg.norm(r).item()
            residual_norms.append(res_norm)
            if res_norm < atol:
                break

            (z, metadata) = (
                self.preconditioner.solve(r)
                if self.preconditioner is not None
                else (r, {})
            )

            preconditioner_metadata.append(metadata)
            rho_cur = torch.dot(r, z)
            if i > 0:
                beta = rho_cur / rho_prev
                fused_mul_add(p, z, beta)
            else:
                p = torch.empty_like(r)
                p[:] = z[:]

            q = self.matrix @ p
            alpha = rho_cur / torch.dot(p, q)
            fused_add(x, p, alpha)
            fused_sub(r, q, alpha)
            rho_prev = rho_cur

            if self.estimate_cond:
                if i > 0:
                    betas.append(beta)
                alphas.append(alpha)

        metadata = {
            "iterations": len(residual_norms) - 1,
            "residual norms": residual_norms,
            "preconditioner metadata": preconditioner_metadata,
        }

        if self.estimate_cond:
            lmat = torch.zeros(
                (len(alphas), len(alphas)), dtype=rhs.dtype, device=rhs.device
            )
            for i in range(len(alphas)):
                lmat[i, i] = 1 / alphas[i]
                if i > 0:
                    lmat[i, i] += betas[i - 1] / alphas[i - 1]
                    lmat[i - 1, i] = lmat[i, i - 1] = (
                        betas[i - 1] ** (1 / 2) / alphas[i - 1]
                    )
            metadata["condition number estimate"] = torch.linalg.cond(lmat, p=2).item()

        if i + 1 == maxiter:
            raise ConvergenceNotAchieved(metadata)

        return x, metadata

    def destroy(self) -> None:
        if self.preconditioner is not None:
            self.preconditioner.destroy()
            self.preconditioner = None
        self.matrix = None


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(perm), device=perm.device, dtype=perm.dtype)
    return inv_perm


class DenseBatchSolver(ABC):
    @abstractmethod
    def setup(self, matrix_batch: torch.Tensor):
        """
        matrix should have shape (batch_size, n, n)
        """
        pass

    @abstractmethod
    def solve(self, rhs_batch: torch.Tensor) -> torch.Tensor:
        pass


class CSRBatch(NamedTuple):
    nrows: torch.Tensor
    nnz: torch.Tensor
    values: torch.Tensor
    col_indices: torch.Tensor
    row_offsets: torch.Tensor


class SparseBatchSolver(ABC):
    @abstractmethod
    def setup(self, matrix_batch: CSRBatch):
        pass

    @abstractmethod
    def solve(self, rhs_batch: torch.Tensor) -> torch.Tensor:
        pass


class Inv(DenseBatchSolver):
    def __init__(self, precision: torch.dtype | None = None):
        self.precision = precision

    def __str__(self):
        return f"Inv({self.precision or ''})"

    def setup(self, matrix_batch: torch.Tensor):
        self.matrix_inv = torch.linalg.inv(matrix_batch).to(
            self.precision
        )  # returns column-major

    def solve(self, rhs_batch: torch.Tensor) -> torch.Tensor:
        return gemv_strided_batched(
            self.matrix_inv,
            rhs_batch.reshape(self.matrix_inv.shape[0], -1),
        )


class LU(DenseBatchSolver):
    def __str__(self):
        return "LU"

    def setup(self, matrix_batch: torch.Tensor):
        self.matrix_lu = torch.linalg.lu_factor(matrix_batch, pivot=False)
        self.n_solvers = matrix_batch.shape[0]

    def solve(self, rhs_batch: torch.Tensor) -> torch.Tensor:
        return torch.linalg.lu_solve(
            *self.matrix_lu, rhs_batch.reshape(self.n_solvers, -1, 1)
        )


class Cholesky(DenseBatchSolver):
    def __str__(self):
        return "Cholesky"

    def setup(self, matrix_batch: torch.Tensor):
        self.matrix_cholesky = torch.linalg.cholesky(matrix_batch)

    def solve(self, rhs_batch: torch.Tensor) -> torch.Tensor:
        return torch.cholesky_solve(
            rhs_batch.reshape(self.matrix_cholesky.shape[0], -1, 1),
            self.matrix_cholesky,
        )


class BatchCUDSS(SparseBatchSolver):
    def __str__(self):
        return "CUDSS"

    def setup(self, matrix_batch: CSRBatch):
        self.solver = cudss.spd_batch_factorize(
            matrix_batch.nrows,
            matrix_batch.nnz,
            matrix_batch.values,
            matrix_batch.col_indices,
            matrix_batch.row_offsets,
        )

    def solve(self, rhs_batch: torch.Tensor) -> torch.Tensor:
        return self.solver.solve(rhs_batch.flatten())


@cuda.jit
def construct_local_solvers_matrices_dense_kernel(
    dofs_per_solver, crow_indices, col_indices, values, output
):
    row = cuda.grid(1)
    if row >= crow_indices.size - 1:
        return

    solver = row // dofs_per_solver
    solver_row = row % dofs_per_solver
    for ind in range(crow_indices[row], crow_indices[row + 1]):
        col = col_indices[ind]
        if col // dofs_per_solver == row // dofs_per_solver:
            solver_col = col % dofs_per_solver
            output[solver, solver_row, solver_col] = values[ind]
        elif col // dofs_per_solver > row // dofs_per_solver:
            break


@cuda.jit
def construct_coarse_solver_matrix_count_kernel(
    dofs_per_solver, solvers_to_coarse, crow_indices, col_indices, output
):
    row = cuda.grid(1)
    if row >= crow_indices.size - 1:
        return

    count = 0
    prev_block = -1
    for ind in range(crow_indices[row], crow_indices[row + 1]):
        col = col_indices[ind]
        col_block = solvers_to_coarse[col // dofs_per_solver]
        if col_block != prev_block:
            count += 1
            prev_block = col_block

    output[row] = count


@cuda.jit
def construct_coarse_solver_matrix_coo_kernel(
    dofs_per_solver,
    solvers_to_coarse,
    crow_indices,
    col_indices,
    values,
    blocks_in_row_scan,
    out_indices,
    out_values,
):
    row = cuda.grid(1)
    if row >= crow_indices.size - 1:
        return

    row_block = solvers_to_coarse[row // dofs_per_solver]
    out_ind = blocks_in_row_scan[row]
    value_sum = 0.0
    prev_block = -1
    for ind in range(crow_indices[row], crow_indices[row + 1]):
        col = col_indices[ind]
        col_block = solvers_to_coarse[col // dofs_per_solver]
        if col_block != prev_block:
            if prev_block != -1:
                out_indices[0, out_ind] = row_block
                out_indices[1, out_ind] = prev_block
                out_values[out_ind] = value_sum
                value_sum = 0
                out_ind += 1
            prev_block = col_block
        value_sum += values[ind]

    if prev_block != -1:
        out_indices[0, out_ind] = row_block
        out_indices[1, out_ind] = prev_block
        out_values[out_ind] = value_sum


@cuda.jit
def construct_local_solvers_matrices_sparse_count_kernel(
    dofs_per_solver, crow_indices, col_indices, output
):
    row = cuda.grid(1)
    if row >= crow_indices.size - 1:
        return

    count = 0
    for ind in range(crow_indices[row], crow_indices[row + 1]):
        col = col_indices[ind]
        if col // dofs_per_solver == row // dofs_per_solver:
            count += 1
        elif col // dofs_per_solver > row // dofs_per_solver:
            break

    output[row] = count


@cuda.jit
def construct_local_solvers_matrices_sparse_batch_csr_kernel(
    dofs_per_solver,
    crow_indices,
    col_indices,
    values,
    per_row_count_scan,
    out_crow_indices,
    out_col_indices,
    out_values,
):
    row = cuda.grid(1)
    if row >= crow_indices.size - 1:
        return

    solver = row // dofs_per_solver
    solver_row = row % dofs_per_solver
    out_ind = per_row_count_scan[row]
    first_solver_ind = per_row_count_scan[solver * dofs_per_solver]

    if solver_row == 0:
        out_crow_indices[solver * (dofs_per_solver + 1)] = out_ind - first_solver_ind

    for ind in range(crow_indices[row], crow_indices[row + 1]):
        col = col_indices[ind]
        if col // dofs_per_solver == row // dofs_per_solver:
            solver_col = col % dofs_per_solver
            out_col_indices[out_ind] = solver_col
            out_values[out_ind] = values[ind]
            out_ind += 1
        elif col // dofs_per_solver > row // dofs_per_solver:
            break

    out_crow_indices[solver * (dofs_per_solver + 1) + solver_row + 1] = (
        out_ind - first_solver_ind
    )


def asm_permutation(
    fine_to_solvers: torch.Tensor,
    solvers_to_coarse: torch.Tensor,
    fine_to_dofs: torch.Tensor,
) -> torch.Tensor:
    n_solvers = solvers_to_coarse.shape[0]
    solver_indices = fine_to_dofs[fine_to_solvers.argsort(stable=True)].reshape(
        n_solvers, -1
    )
    perm = solver_indices[solvers_to_coarse.argsort(stable=True)].flatten()
    return perm


class SchwarzOperator(SparseSolver):
    """
    Assumes every solver has the same number of dofs.
    """

    def __init__(
        self,
        preconditioner_precision: torch.dtype,
        local_solver: DenseBatchSolver | SparseBatchSolver,
        coarse_solver: SparseSolver,
        collect_timings: bool = False,
    ):
        self.preconditioner_precision = preconditioner_precision
        self.local_solver = local_solver
        self.coarse_solver = coarse_solver
        self.collect_timings = collect_timings

    def setup(
        self,
        matrix: torch.Tensor,
        dofs_per_solver: int,
        solvers_per_coarse: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        self.device = matrix.device
        assert matrix.is_sparse_csr

        self.dofs_per_solver = dofs_per_solver
        self.solvers_per_coarse = solvers_per_coarse
        self.n_coarse = solvers_per_coarse.shape[0]

        if self.collect_timings:
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            events[0].record()

        self.solvers_per_coarse_scan = torch.empty(
            self.n_coarse + 1,
            device=self.device,
            dtype=torch.int32,
        )
        self.solvers_per_coarse_scan[0] = 0
        self.solvers_per_coarse_scan[1:] = solvers_per_coarse.cumsum(dim=0)

        self.n_solvers = int(self.solvers_per_coarse_scan[-1].item())

        if self.collect_timings:
            events[1].record()

        collect()
        self.coarse_solver.setup(self._construct_coarse_solver_matrix(matrix))

        if self.collect_timings:
            events[2].record()

        collect()
        if isinstance(self.local_solver, DenseBatchSolver):
            A_i_holder = [self._construct_local_solvers_matrices_dense(matrix)]
        elif isinstance(self.local_solver, SparseBatchSolver):
            A_i_holder = [self._construct_local_solvers_matrices_sparse(matrix)]
        else:
            raise ValueError(f"Unknown local solver type: {type(self.local_solver)}")

        self.local_solver.setup(A_i_holder.pop())

        if self.collect_timings:
            events[3].record()
            torch.cuda.synchronize()

            x_c = torch.rand(
                self.n_coarse, device=self.device, dtype=self.preconditioner_precision
            )
            coarse_solve_time = timeit(lambda: self.coarse_solver.solve(x_c))
            x_i = torch.rand(
                (self.n_solvers, self.dofs_per_solver),
                device=self.device,
                dtype=self.preconditioner_precision,
            )
            local_solve_time = timeit(lambda: self.local_solver.solve(x_i))

            return {
                "preprocessing time": events[0].elapsed_time(events[1]),
                "coarse solver setup time": events[1].elapsed_time(events[2]),
                "local solver setup time": events[2].elapsed_time(events[3]),
                "coarse solver solve time": coarse_solve_time,
                "local solver solve time": local_solve_time,
            }

    def destroy(self) -> None:
        self.coarse_solver.destroy()

    def _construct_local_solvers_matrices_dense(self, Ap: torch.Tensor) -> torch.Tensor:
        A_i = torch.zeros(
            (self.n_solvers, self.dofs_per_solver, self.dofs_per_solver),
            device=self.device,
            dtype=self.preconditioner_precision or Ap.values().dtype,
        )

        thread_block_size = 32
        grid_size = (Ap.size(0) + thread_block_size - 1) // thread_block_size
        construct_local_solvers_matrices_dense_kernel[grid_size, thread_block_size](
            self.dofs_per_solver, Ap.crow_indices(), Ap.col_indices(), Ap.values(), A_i
        )

        return A_i

    def _construct_local_solvers_matrices_sparse(self, Ap: torch.Tensor) -> CSRBatch:
        nrows = torch.full(
            (self.n_solvers,),
            self.dofs_per_solver,
            dtype=torch.int32,
            device=self.device,
        )

        per_row_count = torch.empty(
            Ap.shape[0] + 1,
            device=self.device,
            dtype=torch.int64,
        )
        thread_block_size = 32
        grid_size = (Ap.shape[0] + thread_block_size - 1) // thread_block_size
        construct_local_solvers_matrices_sparse_count_kernel[
            grid_size, thread_block_size
        ](self.dofs_per_solver, Ap.crow_indices(), Ap.col_indices(), per_row_count[1:])

        nnz = per_row_count[1:].reshape(self.n_solvers, -1).sum(dim=1)

        per_row_count[0] = 0
        per_row_count.cumsum_(dim=0)

        total_count = int(per_row_count[-1].item())
        values = torch.empty(
            total_count,
            device=self.device,
            dtype=self.preconditioner_precision or Ap.values().dtype,
        )
        col_indices = torch.empty(
            total_count,
            device=self.device,
            dtype=torch.int32,
        )
        row_offsets = torch.empty(
            Ap.shape[0] + self.n_solvers,
            device=self.device,
            dtype=torch.int32,
        )
        construct_local_solvers_matrices_sparse_batch_csr_kernel[
            grid_size, thread_block_size
        ](
            self.dofs_per_solver,
            Ap.crow_indices(),
            Ap.col_indices(),
            Ap.values(),
            per_row_count,
            row_offsets,
            col_indices,
            values,
        )

        return CSRBatch(
            nrows=nrows,
            nnz=nnz,
            values=values,
            col_indices=col_indices,
            row_offsets=row_offsets,
        )

    def _construct_coarse_solver_matrix(self, Ap: torch.Tensor) -> torch.Tensor:
        solvers_to_coarse = torch.arange(
            self.n_coarse, device=self.device
        ).repeat_interleave(self.solvers_per_coarse)
        blocks_in_row_scan = torch.empty(
            Ap.shape[0] + 1, device=self.device, dtype=torch.int64
        )

        thread_block_size = 32
        grid_size = (Ap.shape[0] + thread_block_size - 1) // thread_block_size
        construct_coarse_solver_matrix_count_kernel[grid_size, thread_block_size](
            self.dofs_per_solver,
            solvers_to_coarse,
            Ap.crow_indices(),
            Ap.col_indices(),
            blocks_in_row_scan[1:],
        )

        blocks_in_row_scan[0] = 0
        blocks_in_row_scan.cumsum_(dim=0)

        total_blocks = int(blocks_in_row_scan[-1].item())
        indices = torch.empty(
            (2, total_blocks),
            device=self.device,
            dtype=torch.int64,
        )
        values = torch.empty(
            total_blocks,
            device=self.device,
            dtype=self.preconditioner_precision or Ap.values().dtype,
        )
        construct_coarse_solver_matrix_coo_kernel[grid_size, thread_block_size](
            self.dofs_per_solver,
            solvers_to_coarse,
            Ap.crow_indices(),
            Ap.col_indices(),
            Ap.values(),
            blocks_in_row_scan,
            indices,
            values,
        )

        del solvers_to_coarse, blocks_in_row_scan
        collect()

        return torch.sparse_coo_tensor(
            indices,
            values,
            size=(self.n_coarse, self.n_coarse),
        ).to_sparse_csr()


class AdditiveSchwarz(SchwarzOperator):
    def __init__(
        self,
        preconditioner_precision: torch.dtype,
        local_solver: DenseBatchSolver | SparseBatchSolver,
        coarse_solver: SparseSolver,
        number_of_dofs_per_coarse_is_const: bool = False,
        collect_timings: bool = False,
    ):
        super().__init__(
            preconditioner_precision,
            local_solver,
            coarse_solver,
            collect_timings,
        )
        self.number_of_dofs_per_coarse_is_const = number_of_dofs_per_coarse_is_const

    def __str__(self):
        return f"AdditiveSchwarz({self.preconditioner_precision}, {self.local_solver}, {self.coarse_solver}, const_dofs_per_coarse={self.number_of_dofs_per_coarse_is_const})"

    def setup(
        self,
        matrix: torch.Tensor,
        dofs_per_solver: int,
        solvers_per_coarse: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        timings = super().setup(
            matrix, dofs_per_solver, solvers_per_coarse, *args, **kwargs
        )
        if self.number_of_dofs_per_coarse_is_const:
            self.dofs_per_coarse = dofs_per_solver * int(
                self.solvers_per_coarse[0].item()
            )
        return timings

    @torch.compile
    def solve(self, rhs: torch.Tensor) -> tuple[torch.Tensor, Metadata]:
        x_lower_precision = rhs.to(self.preconditioner_precision or rhs.dtype)
        x_i = x_lower_precision.reshape(self.n_solvers, -1)
        y_i = self.local_solver.solve(x_i)
        y = y_i.flatten()
        x_solvers = x_lower_precision.reshape(self.n_solvers, -1).sum(dim=1)
        x_c = torch.segment_reduce(
            x_solvers, reduce="sum", offsets=self.solvers_per_coarse_scan
        )
        y_c, _ = self.coarse_solver.solve(x_c)
        y_solvers = y_c.repeat_interleave(
            self.solvers_per_coarse, output_size=self.n_solvers
        )
        y += y_solvers.repeat_interleave(self.dofs_per_solver, output_size=y.shape[0])
        return y.to(rhs.dtype), {}


class HybridSchwarz(SchwarzOperator):
    def __init__(
        self,
        preconditioner_precision: torch.dtype,
        local_solver: DenseBatchSolver | SparseBatchSolver,
        coarse_solver: SparseSolver,
        collect_timings: bool = False,
    ):
        super().__init__(
            preconditioner_precision,
            local_solver,
            coarse_solver,
            collect_timings,
        )

    def __str__(self):
        return f"HybridSchwarz({self.preconditioner_precision}, {self.local_solver}, {self.coarse_solver})"

    def setup(
        self,
        matrix: torch.Tensor,
        dofs_per_solver: int,
        solvers_per_coarse: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        timings = super().setup(
            matrix, dofs_per_solver, solvers_per_coarse, *args, **kwargs
        )
        self.R0A = self._construct_R0A_matrix(matrix)
        return timings

    def T0(self, rhs: torch.Tensor) -> torch.Tensor:
        x_lower_precision = rhs.to(self.preconditioner_precision or rhs.dtype)
        x_c = x_lower_precision.reshape(self.n_solvers, -1).sum(dim=1)
        x_c = torch.segment_reduce(
            x_c, reduce="sum", offsets=self.solvers_per_coarse_scan
        )
        y_c, _ = self.coarse_solver.solve(x_c)
        y_solvers = y_c.repeat_interleave(
            self.solvers_per_coarse, output_size=self.n_solvers
        )
        y = y_solvers.repeat_interleave(self.dofs_per_solver, output_size=rhs.shape[0])
        return y.to(rhs.dtype)

    @torch.compile
    def solve(self, rhs: torch.Tensor) -> Tuple[torch.Tensor, Metadata]:
        x_lower_precision = rhs.to(self.preconditioner_precision or rhs.dtype)
        res = self.local_solver.solve(
            x_lower_precision.reshape(self.n_solvers, -1)
        ).flatten()
        z = self.R0A @ res
        y_c, _ = self.coarse_solver.solve(z)
        y_solvers = y_c.repeat_interleave(
            self.solvers_per_coarse, output_size=self.n_solvers
        )
        res -= y_solvers.repeat_interleave(
            self.dofs_per_solver, output_size=res.shape[0]
        )
        return res.to(rhs.dtype), {}

    def _construct_R0A_matrix(self, Ap: torch.Tensor) -> torch.Tensor:
        return gather_csr(
            sort_csr(
                torch.sparse_csr_tensor(
                    crow_indices=Ap.crow_indices()[
                        self.solvers_per_coarse_scan * self.dofs_per_solver
                    ],
                    col_indices=Ap.col_indices().clone(),
                    values=(
                        Ap.values().clone()
                        if self.preconditioner_precision == Ap.dtype
                        else Ap.values().to(self.preconditioner_precision)
                    ),
                    size=(self.n_coarse, Ap.shape[1]),
                )
            )
        )


class AMGX(SparseSolver):
    _configs = {
        # AMG_CLASSICAL_L1_TRUNC
        "L1_TRUNC": {
            "config_version": 2,
            "solver": {
                "scope": "main",
                "interpolator": "D2",
                "solver": "AMG",
                "interp_max_elements": 4,
                "smoother": {
                    "relaxation_factor": 1,
                    "scope": "jacobi",
                    "solver": "JACOBI_L1",
                },
                "presweeps": 1,
                "coarsest_sweeps": 1,
                "coarse_solver": "NOSOLVER",
                "max_iters": 1,
                "max_row_sum": 0.9,
                "max_levels": 50,
                "postsweeps": 1,
                "cycle": "V",
                "monitor_residual": 0,
            },
        },
        "CG_L1_TRUNC": {
            "config_version": 2,
            "solver": {
                "scope": "main",
                "solver": "PCG",
                "max_iters": 2000,
                "monitor_residual": 1,
                "convergence": "RELATIVE_INI",
                "tolerance": 1e-9,
                "norm": "L2",
                "preconditioner": {
                    "scope": "amg_solver",
                    "interpolator": "D2",
                    "solver": "AMG",
                    "interp_max_elements": 4,
                    "smoother": {
                        "relaxation_factor": 1,
                        "scope": "jacobi",
                        "solver": "JACOBI_L1",
                    },
                    "presweeps": 1,
                    "coarsest_sweeps": 1,
                    "coarse_solver": "NOSOLVER",
                    "max_iters": 1,
                    "max_row_sum": 0.9,
                    "max_levels": 50,
                    "postsweeps": 1,
                    "cycle": "V",
                },
            },
        },
        # AMG_CLASSICAL_AGGRESSIVE_L1
        "AGGRESIVE_L1": {
            "config_version": 2,
            "solver": {
                "scope": "main",
                "interpolator": "D2",
                "aggressive_levels": 1,
                "solver": "AMG",
                "smoother": {
                    "relaxation_factor": 1,
                    "scope": "jacobi",
                    "solver": "JACOBI_L1",
                },
                "presweeps": 1,
                "selector": "PMIS",
                "coarsest_sweeps": 1,
                "coarse_solver": "NOSOLVER",
                "max_iters": 1,
                "max_row_sum": 0.9,
                "strength_threshold": 0.25,
                "min_coarse_rows": 2,
                "max_levels": 50,
                "cycle": "V",
                "postsweeps": 1,
            },
        },
        # AMG_CLASSICAL_PMIS
        # FGMRES_CLASSICAL_AGGRESSIVE_PMIS
        # AMG_CLASSICAL_AGGRESSIVE_L1_TRUNC
        "AGGRESSIVE_L1_TRUNC": {
            "config_version": 2,
            "solver": {
                "interpolator": "D2",
                "solver": "AMG",
                "aggressive_levels": 1,
                "interp_max_elements": 4,
                "smoother": {
                    "relaxation_factor": 1,
                    "scope": "jacobi",
                    "solver": "JACOBI_L1",
                },
                "presweeps": 2,
                "selector": "PMIS",
                "coarsest_sweeps": 2,
                "coarse_solver": "NOSOLVER",
                "max_iters": 1,
                "max_row_sum": 0.9,
                "strength_threshold": 0.25,
                "min_coarse_rows": 2,
                "max_levels": 50,
                "cycle": "V",
                "postsweeps": 2,
            },
        },
        # AMG_CLASSICAL_AGGRESSIVE_CHEB_L1_TRUNC
        "AGGRESSIVE_CHEB_L1_TRUNC": {
            "config_version": 2,
            "solver": {
                "scope": "main",
                "interpolator": "D2",
                "solver": "AMG",
                "max_levels": 50,
                "selector": "PMIS",
                "cycle": "V",
                "presweeps": 0,
                "postsweeps": 3,
                "coarsest_sweeps": 2,
                "min_coarse_rows": 2,
                "coarse_solver": "NOSOLVER",
                "max_iters": 1,
                "max_row_sum": 0.9,
                "strength_threshold": 0.25,
                "error_scaling": 3,
                "aggressive_levels": 1,
                "interp_max_elements": 4,
                "smoother": {
                    "relaxation_factor": 0.91,
                    "scope": "jacobi",
                    "solver": "CHEBYSHEV",
                    "preconditioner": {"solver": "JACOBI_L1", "max_iters": 1},
                    "chebyshev_polynomial_order": 2,
                    "chebyshev_lambda_estimate_mode": 2,
                },
            },
        },
    }
    preconditioner_config_names = [
        config_name
        for config_name in _configs.keys()
        if not config_name.startswith("CG_")
    ]

    def __init__(self, config_name: str, precision: torch.dtype | None = None):
        self.config_name = config_name
        self.precision = precision

    def __str__(self):
        return f"AMGX({self.config_name}{', ' + str(self.precision) or ''})"

    def setup(self, matrix: torch.Tensor, *args, **kwargs) -> None:
        # Aggressive PMIS selector not implemented on host
        assert matrix.is_cuda

        self.precision = matrix.dtype if self.precision is None else self.precision

        self.mode = "d"
        if self.precision == torch.float64:
            self.mode += "DD"
        elif self.precision == torch.float32:
            self.mode += "FF"
        else:
            raise ValueError(f"Unsupported dtype: {self.precision}")
        self.mode += "I"

        self.config = pyamgx.Config().create_from_dict(self._configs[self.config_name])
        self.rsc = pyamgx.Resources().create_simple(self.config)
        self.solver = pyamgx.Solver().create(self.rsc, self.config, mode=self.mode)
        self.matrix = pyamgx.Matrix().create(self.rsc, mode=self.mode)
        self.b = pyamgx.Vector().create(self.rsc, mode=self.mode)
        self.x = pyamgx.Vector().create(self.rsc, mode=self.mode)
        self.x.set_zero(matrix.shape[1], block_dim=1)

        self.matrix.upload(
            row_ptrs=matrix.crow_indices().to(torch.int32),
            col_indices=matrix.col_indices().to(torch.int32),
            data=matrix.values().to(self.precision),
        )
        self.solver.setup(self.matrix)

    def solve(self, rhs: torch.Tensor) -> Tuple[torch.Tensor, Metadata]:
        assert rhs.is_cuda

        rhs_lower = rhs.to(self.precision)

        self.b.upload_raw(rhs_lower.data_ptr(), rhs.shape[0])
        self.solver.solve(self.b, self.x, zero_initial_guess=True)
        x_lower = torch.empty_like(rhs_lower)
        self.x.download_raw(x_lower.data_ptr())

        return x_lower.to(rhs.dtype), {}

    def destroy(self) -> None:
        self.matrix.destroy()
        self.b.destroy()
        self.x.destroy()
        self.solver.destroy()
        self.rsc.destroy()
        self.config.destroy()
