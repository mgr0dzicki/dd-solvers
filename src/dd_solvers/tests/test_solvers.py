import pytest

import copy
import numpy as np
import torch

from dd_solvers import (
    asm_permutation,
    inverse_permutation,
    permute_csr_tensor,
    AMGX,
    CG,
    CUDSS,
    LU,
    BatchCUDSS,
    Cholesky,
    SparseSolver,
    AdditiveSchwarz,
    HybridSchwarz,
    Inv,
)

# fmt: off

form_matrix = torch.sparse_csr_tensor(
    values=torch.tensor(
        [ 3.31572972,  1.1923425 ,  1.23906285, -1.08129316, -0.13516164,
         -0.81096987, -1.49832545, -0.42809299, -0.32106974,  1.1923425 ,
          5.25258945,  1.37740998, -0.13516164, -1.08129316, -0.81096987,
         -0.32106974, -0.32106974,  0.        ,  1.23906285,  1.37740998,
          4.29137116, -0.81096987, -0.81096987,  0.        , -0.42809299,
         -0.85618597, -0.32106974, -1.08129316, -0.13516164, -0.81096987,
          4.26856494,  1.52747385,  1.46368244, -2.284494  , -0.65271257,
         -0.48953443, -0.13516164, -1.08129316, -0.81096987,  1.52747385,
          6.1970339 ,  1.60143022, -0.48953443, -0.48953443,  0.        ,
         -0.81096987, -0.81096987,  0.        ,  1.46368244,  1.60143022,
          5.51838811, -0.65271257, -1.30542514, -0.48953443, -1.49832545,
         -0.32106974, -0.42809299,  2.82702916,  0.99753743,  0.98542159,
         -0.75925926, -0.56944444, -0.09490741, -0.42809299, -0.32106974,
         -0.85618597,  0.99753743,  3.51359338,  1.080329  , -0.56944444,
          0.        , -0.56944444, -0.32106974,  0.        , -0.32106974,
          0.98542159,  1.080329  ,  3.98611111, -0.09490741, -0.56944444,
         -0.75925926, -2.284494  , -0.48953443, -0.65271257,  4.77986437,
          1.72215702,  1.73721961, -1.42592593, -1.06944444, -0.17824074,
         -0.65271257, -0.48953443, -1.30542514,  1.72215702,  6.29616588,
          1.91546035, -1.06944444,  0.        , -1.06944444, -0.48953443,
          0.        , -0.48953443,  1.73721961,  1.91546035,  7.48611111,
         -0.17824074, -1.06944444, -1.42592593, -0.75925926, -0.56944444,
         -0.09490741,  2.82702916,  0.99753743,  0.98542159, -1.49832545,
         -0.32106974, -0.42809299, -0.56944444,  0.        , -0.56944444,
          0.99753743,  3.51359338,  1.080329  , -0.42809299, -0.32106974,
         -0.85618597, -0.09490741, -0.56944444, -0.75925926,  0.98542159,
          1.080329  ,  3.98611111, -0.32106974,  0.        , -0.32106974,
         -1.42592593, -1.06944444, -0.17824074,  4.77986437,  1.72215702,
          1.73721961, -2.284494  , -0.48953443, -0.65271257, -1.06944444,
          0.        , -1.06944444,  1.72215702,  6.29616588,  1.91546035,
         -0.65271257, -0.48953443, -1.30542514, -0.17824074, -1.06944444,
         -1.42592593,  1.73721961,  1.91546035,  7.48611111, -0.48953443,
          0.        , -0.48953443, -1.49832545, -0.42809299, -0.32106974,
          3.31572972,  1.1923425 ,  1.23906285, -1.08129316, -0.13516164,
         -0.81096987, -0.32106974, -0.32106974,  0.        ,  1.1923425 ,
          5.25258945,  1.37740998, -0.13516164, -1.08129316, -0.81096987,
         -0.42809299, -0.85618597, -0.32106974,  1.23906285,  1.37740998,
          4.29137116, -0.81096987, -0.81096987,  0.        , -2.284494  ,
         -0.65271257, -0.48953443, -1.08129316, -0.13516164, -0.81096987,
          4.26856494,  1.52747385,  1.46368244, -0.48953443, -0.48953443,
          0.        , -0.13516164, -1.08129316, -0.81096987,  1.52747385,
          6.1970339 ,  1.60143022, -0.65271257, -1.30542514, -0.48953443,
         -0.81096987, -0.81096987,  0.        ,  1.46368244,  1.60143022,
          5.51838811]
    ),
    col_indices=torch.tensor(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  0,  1,  2,  3,  4,  5,  6,  7,
          8,  0,  1,  2,  3,  4,  5,  6,  7,  8,  0,  1,  2,  3,  4,  5,  9,
         10, 11,  0,  1,  2,  3,  4,  5,  9, 10, 11,  0,  1,  2,  3,  4,  5,
          9, 10, 11,  0,  1,  2,  6,  7,  8, 12, 13, 14,  0,  1,  2,  6,  7,
          8, 12, 13, 14,  0,  1,  2,  6,  7,  8, 12, 13, 14,  3,  4,  5,  9,
         10, 11, 15, 16, 17,  3,  4,  5,  9, 10, 11, 15, 16, 17,  3,  4,  5,
          9, 10, 11, 15, 16, 17,  6,  7,  8, 12, 13, 14, 18, 19, 20,  6,  7,
          8, 12, 13, 14, 18, 19, 20,  6,  7,  8, 12, 13, 14, 18, 19, 20,  9,
         10, 11, 15, 16, 17, 21, 22, 23,  9, 10, 11, 15, 16, 17, 21, 22, 23,
          9, 10, 11, 15, 16, 17, 21, 22, 23, 12, 13, 14, 18, 19, 20, 21, 22,
         23, 12, 13, 14, 18, 19, 20, 21, 22, 23, 12, 13, 14, 18, 19, 20, 21,
         22, 23, 15, 16, 17, 18, 19, 20, 21, 22, 23, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        dtype=torch.int32,
    ),
    crow_indices=torch.tensor(
        [  0,   9,  18,  27,  36,  45,  54,  63,  72,  81,  90,  99, 108,
         117, 126, 135, 144, 153, 162, 171, 180, 189, 198, 207, 216],
        dtype=torch.int32,
    ),
    size=torch.Size((24, 24)),
)

load_vector = torch.tensor(
    [1.73900805e-16, 1.09778448e-16, 1.55434540e-16, 4.89013992e-16,
     4.24891636e-16, 7.85660915e-16, 1.28244713e-16, 6.41223565e-17,
     6.41223565e-17, 9.78027985e-16, 1.04215034e-15, 1.63544419e-15,
     1.28244713e-16, 6.41223565e-17, 6.41223565e-17, 9.78027985e-16,
     1.04215034e-15, 1.63544419e-15, 1.73900805e-16, 1.09778448e-16,
     1.55434540e-16, 4.89013992e-16, 4.24891636e-16, 7.85660915e-16]
)

# fmt: on

fine_to_solvers = torch.tensor([3, 3, 2, 1, 2, 1, 0, 0], dtype=torch.int32)
solvers_to_coarse = torch.tensor([0, 0, 1, 2], dtype=torch.int32)


device = torch.device("cuda")


# fmt: off
@pytest.mark.parametrize(
    "solver,problem_dtype",
    [
        (CUDSS(), torch.float32),
        (CUDSS(), torch.float64),
        (AMGX("CG_L1_TRUNC"), torch.float64),
        (CG(bsr_matmul=False), torch.float32),
        (CG(bsr_matmul=True), torch.float64),
        (CG(AMGX("AGGRESIVE_L1")), torch.float64),
        (CG(AMGX("L1_TRUNC", torch.float32)), torch.float64),
        (CG(AMGX("AGGRESSIVE_CHEB_L1_TRUNC")), torch.float32),
        (CG(AdditiveSchwarz(torch.float64, Inv(), CUDSS())), torch.float64),
        (CG(AdditiveSchwarz(torch.float32, Inv(), CUDSS())), torch.float64),
        (CG(AdditiveSchwarz(torch.float32, Inv(), CUDSS())), torch.float32),
        (CG(AdditiveSchwarz(torch.float32, Inv(torch.float16), CUDSS())), torch.float32),
        (CG(AdditiveSchwarz(torch.float64, LU(), CUDSS())), torch.float64),
        (CG(AdditiveSchwarz(torch.float32, LU(), CUDSS())), torch.float64),
        (CG(AdditiveSchwarz(torch.float32, LU(), CUDSS())), torch.float32),
        (CG(AdditiveSchwarz(torch.float64, Cholesky(), CUDSS())), torch.float64),
        (CG(AdditiveSchwarz(torch.float32, Cholesky(), CUDSS())), torch.float64),
        (CG(AdditiveSchwarz(torch.float32, Cholesky(), CUDSS())), torch.float32),
        (CG(AdditiveSchwarz(torch.float64, BatchCUDSS(), CUDSS())), torch.float64),
        (CG(AdditiveSchwarz(torch.float32, BatchCUDSS(), CUDSS())), torch.float64),
        (CG(AdditiveSchwarz(torch.float32, BatchCUDSS(), CUDSS())), torch.float32),
        (CG(AdditiveSchwarz(torch.float64, Inv(), AMGX("AGGRESSIVE_L1_TRUNC"))), torch.float64),
        (CG(HybridSchwarz(torch.float64, Inv(), CUDSS())), torch.float64),
        (CG(HybridSchwarz(torch.float64, Inv(torch.float32), CUDSS())), torch.float64),
        (CG(HybridSchwarz(torch.float64, Inv(torch.float16), CUDSS())), torch.float64),
        (CG(HybridSchwarz(torch.float64, Inv(torch.bfloat16), CUDSS())), torch.float64),
        (CG(HybridSchwarz(torch.float32, Inv(), CUDSS()), rtol=1e-6), torch.float32),
        (CG(HybridSchwarz(torch.float32, Inv(), CUDSS()), rtol=1e-6), torch.float64),
    ],
)
# fmt: on
def test_sparse_solver(solver: SparseSolver, problem_dtype: np.dtype):
    solver = copy.deepcopy(solver)
    matrix = form_matrix.to(device=device, dtype=problem_dtype)
    rhs = load_vector.to(device=device, dtype=problem_dtype)
    fine_to_dofs = torch.arange(24, device=device, dtype=torch.int32).reshape(8, 3)

    n_coarse = solvers_to_coarse.max().item() + 1
    dofs_per_solver = 6
    solvers_per_coarse = solvers_to_coarse.bincount(minlength=n_coarse)

    perm = asm_permutation(
        fine_to_solvers=fine_to_solvers,
        solvers_to_coarse=solvers_to_coarse,
        fine_to_dofs=fine_to_dofs,
    ).to(device)
    inv_perm = inverse_permutation(perm)
    matrix = permute_csr_tensor(matrix, inv_perm)

    solver.setup(
        matrix=matrix,
        dofs_per_solver=dofs_per_solver,
        solvers_per_coarse=solvers_per_coarse.to(device),
        block_size=fine_to_dofs.shape[1],
    )
    x, _metadata = solver.solve(rhs[perm])
    solver.destroy()

    assert torch.linalg.norm(matrix @ x[inv_perm] - rhs) < 1e-10


def test_cg_condition_number_estimate():
    solver = CG(preconditioner=None, rtol=1e-9, estimate_cond=True, bsr_matmul=False)
    solver.setup(torch.diag(torch.tensor([0.5, 4, 21])))
    _, metadata = solver.solve(torch.ones(3))
    cond_estimate = metadata["condition number estimate"]
    assert np.isclose(cond_estimate, 42.0)
