import pytest
import torch
import warnings
import gc
from dd_solvers import cudss

device = torch.device("cuda")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # 2 0 2
    # 0 2 0
    # 2 0 3
    spd_3x3 = torch.sparse_csr_tensor(
        crow_indices=torch.tensor([0, 2, 3, 5], dtype=torch.int32),
        col_indices=torch.tensor([0, 2, 1, 0, 2], dtype=torch.int32),
        values=torch.tensor([2, 2, 2, 2, 3], dtype=torch.float32),
        size=(3, 3),
    ).to(device)

    # 2 1
    # 1 1
    spd_2x2 = torch.sparse_csr_tensor(
        crow_indices=torch.tensor([0, 2, 4], dtype=torch.int32),
        col_indices=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        values=torch.tensor([2, 1, 1, 1], dtype=torch.float32),
        size=(2, 2),
    ).to(device)


@pytest.mark.parametrize("value_type", [torch.float32, torch.float64])
@pytest.mark.parametrize("index_type", [torch.int32, torch.int64])
def test_spd_factorize(value_type: torch.dtype, index_type: torch.dtype):
    A = torch.sparse_csr_tensor(
        crow_indices=spd_3x3.crow_indices().to(dtype=index_type, copy=True),
        col_indices=spd_3x3.col_indices().to(dtype=index_type, copy=True),
        values=spd_3x3.values().to(dtype=value_type, copy=True),
        size=spd_3x3.size(),
    )
    solver = cudss.spd_factorize(A)

    del A
    gc.collect()
    torch.cuda.empty_cache()

    x1 = torch.tensor([1, -1, 0], device=device, dtype=value_type)
    x2 = torch.tensor([1, 2, 1], device=device, dtype=value_type)
    b1 = torch.tensor([2, -2, 2], device=device, dtype=value_type)
    b2 = torch.tensor([4, 4, 5], device=device, dtype=value_type)

    x1_hat = solver.solve(b1)
    x2_hat = solver.solve(b2)

    assert torch.allclose(x1, x1_hat, atol=1e-6)
    assert x1_hat.dtype == x1.dtype
    assert torch.allclose(x2, x2_hat, atol=1e-6)
    assert x2_hat.dtype == x2.dtype


@pytest.mark.parametrize("value_type", [torch.float32, torch.float64])
@pytest.mark.parametrize("index_type", [torch.int32, torch.int64])
def test_spd_batch_factorize(value_type: torch.dtype, index_type: torch.dtype):
    nrows = torch.tensor([spd_3x3.size(0), spd_2x2.size(0)], dtype=index_type)
    nnz = torch.tensor([spd_3x3._nnz(), spd_2x2._nnz()], dtype=index_type)
    values = torch.cat([spd_3x3.values(), spd_2x2.values()]).to(value_type)
    col_indices = torch.cat([spd_3x3.col_indices(), spd_2x2.col_indices()]).to(
        index_type
    )
    row_offsets = torch.cat([spd_3x3.crow_indices(), spd_2x2.crow_indices()]).to(
        index_type
    )

    solver = cudss.spd_batch_factorize(nrows, nnz, values, col_indices, row_offsets)

    del nrows, nnz, values, col_indices, row_offsets
    gc.collect()
    torch.cuda.empty_cache()

    x = torch.tensor([1, -1, 1, -1, 1], device=device, dtype=value_type)
    b = torch.tensor([4, -2, 5, -1, 0], device=device, dtype=value_type)

    x_hat = solver.solve(b)
    assert torch.allclose(x, x_hat, atol=1e-6)
