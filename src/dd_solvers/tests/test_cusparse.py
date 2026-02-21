import pytest
import torch
from dd_solvers.cusparse import sort_csr, gather_csr

device = torch.device("cuda")


@pytest.mark.parametrize("ind_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_parts", [1, 2, 3, 4, None])
def test_sort_csr(ind_dtype: torch.dtype, dtype: torch.dtype, num_parts: int):
    # 1 2 0 0
    # 0 3 0 0
    # 4 0 5 0
    # 6 7 8 9
    matrix_csr = torch.sparse_csr_tensor(
        crow_indices=torch.tensor([0, 2, 3, 5, 9], dtype=ind_dtype),
        col_indices=torch.tensor([1, 0, 1, 0, 2, 2, 3, 1, 0], dtype=ind_dtype),
        values=torch.tensor([2, 1, 3, 4, 5, 8, 9, 7, 6], dtype=dtype),
    ).to(device=device)

    sorted_matrix = sort_csr(matrix_csr, num_parts=num_parts)
    assert torch.equal(
        sorted_matrix.crow_indices(),
        torch.tensor([0, 2, 3, 5, 9], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        sorted_matrix.col_indices(),
        torch.tensor([0, 1, 1, 0, 2, 0, 1, 2, 3], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        sorted_matrix.values(),
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=dtype, device=device),
    )


def test_gather_csr_simple():
    matrix_csr = torch.sparse_csr_tensor(
        crow_indices=torch.tensor([0, 3, 6], dtype=torch.int32),
        col_indices=torch.tensor([0, 2, 2, 0, 0, 1], dtype=torch.int32),
        values=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32),
    ).to(device=device)

    matrix_gathered = gather_csr(matrix_csr)
    assert torch.equal(
        matrix_gathered.crow_indices(),
        torch.tensor([0, 2, 4], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        matrix_gathered.col_indices(),
        torch.tensor([0, 2, 0, 1], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        matrix_gathered.values(),
        torch.tensor([1.0, 5.0, 9.0, 6.0], dtype=torch.float32, device=device),
    )


def test_gather_csr_empty_row():
    matrix_csr = torch.sparse_csr_tensor(
        crow_indices=torch.tensor([0, 2, 2, 5], dtype=torch.int32),
        col_indices=torch.tensor([0, 1, 0, 0, 1], dtype=torch.int32),
        values=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32),
    ).to(device=device)

    matrix_gathered = gather_csr(matrix_csr)
    assert torch.equal(
        matrix_gathered.crow_indices(),
        torch.tensor([0, 2, 2, 4], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        matrix_gathered.col_indices(),
        torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        matrix_gathered.values(),
        torch.tensor([1.0, 2.0, 7.0, 5.0], dtype=torch.float32, device=device),
    )
