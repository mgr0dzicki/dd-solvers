import torch

torch.ops.dd_solvers_cudss.init()


__all__ = ["CudssSolver", "spd_factorize", "CudssBatchSolver", "spd_batch_factorize"]


class CudssSolver:
    def __init__(self, cudss_solver):
        self._cudss_solver = cudss_solver

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        return self._cudss_solver.solve(rhs)


def spd_factorize(A: torch.Tensor) -> CudssSolver:
    return CudssSolver(torch.ops.dd_solvers_cudss.spd_factorize(A))


class CudssBatchSolver:
    def __init__(self, cudss_batch_solver):
        self._cudss_batch_solver = cudss_batch_solver

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        return self._cudss_batch_solver.solve(rhs)


def spd_batch_factorize(
    nrows: torch.Tensor,
    nnz: torch.Tensor,
    values: torch.Tensor,
    col_indices: torch.Tensor,
    row_start: torch.Tensor,
):
    return CudssBatchSolver(
        torch.ops.dd_solvers_cudss.spd_batch_factorize(
            nrows,
            nnz,
            values,
            col_indices,
            row_start,
        )
    )
