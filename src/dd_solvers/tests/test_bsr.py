import pytest
from dd_solvers.bsr import FastBSR
from dd_solvers.cusparse import csr_to_bsr

import torch

device = torch.device("cuda")

matrix = torch.sparse_bsr_tensor(
    crow_indices=torch.tensor([0, 3, 6, 9, 12, 15, 18, 21, 24]),
    col_indices=torch.tensor(
        [0, 1, 2, 0, 1, 3, 0, 2, 4, 1, 3, 5, 2, 4, 6, 3, 5, 7, 4, 6, 7, 5, 6, 7]
    ),
    values=torch.tensor(
        [
            [
                [3.3157, 1.1923, 1.2391],
                [1.1923, 5.2526, 1.3774],
                [1.2391, 1.3774, 4.2914],
            ],
            [
                [-1.0813, -0.1352, -0.8110],
                [-0.1352, -1.0813, -0.8110],
                [-0.8110, -0.8110, 0.0000],
            ],
            [
                [-1.4983, -0.4281, -0.3211],
                [-0.3211, -0.3211, 0.0000],
                [-0.4281, -0.8562, -0.3211],
            ],
            [
                [-1.0813, -0.1352, -0.8110],
                [-0.1352, -1.0813, -0.8110],
                [-0.8110, -0.8110, 0.0000],
            ],
            [
                [4.2686, 1.5275, 1.4637],
                [1.5275, 6.1970, 1.6014],
                [1.4637, 1.6014, 5.5184],
            ],
            [
                [-2.2845, -0.6527, -0.4895],
                [-0.4895, -0.4895, 0.0000],
                [-0.6527, -1.3054, -0.4895],
            ],
            [
                [-1.4983, -0.3211, -0.4281],
                [-0.4281, -0.3211, -0.8562],
                [-0.3211, 0.0000, -0.3211],
            ],
            [
                [2.8270, 0.9975, 0.9854],
                [0.9975, 3.5136, 1.0803],
                [0.9854, 1.0803, 3.9861],
            ],
            [
                [-0.7593, -0.5694, -0.0949],
                [-0.5694, 0.0000, -0.5694],
                [-0.0949, -0.5694, -0.7593],
            ],
            [
                [-2.2845, -0.4895, -0.6527],
                [-0.6527, -0.4895, -1.3054],
                [-0.4895, 0.0000, -0.4895],
            ],
            [
                [4.7799, 1.7222, 1.7372],
                [1.7222, 6.2962, 1.9155],
                [1.7372, 1.9155, 7.4861],
            ],
            [
                [-1.4259, -1.0694, -0.1782],
                [-1.0694, 0.0000, -1.0694],
                [-0.1782, -1.0694, -1.4259],
            ],
            [
                [-0.7593, -0.5694, -0.0949],
                [-0.5694, 0.0000, -0.5694],
                [-0.0949, -0.5694, -0.7593],
            ],
            [
                [2.8270, 0.9975, 0.9854],
                [0.9975, 3.5136, 1.0803],
                [0.9854, 1.0803, 3.9861],
            ],
            [
                [-1.4983, -0.3211, -0.4281],
                [-0.4281, -0.3211, -0.8562],
                [-0.3211, 0.0000, -0.3211],
            ],
            [
                [-1.4259, -1.0694, -0.1782],
                [-1.0694, 0.0000, -1.0694],
                [-0.1782, -1.0694, -1.4259],
            ],
            [
                [4.7799, 1.7222, 1.7372],
                [1.7222, 6.2962, 1.9155],
                [1.7372, 1.9155, 7.4861],
            ],
            [
                [-2.2845, -0.4895, -0.6527],
                [-0.6527, -0.4895, -1.3054],
                [-0.4895, 0.0000, -0.4895],
            ],
            [
                [-1.4983, -0.4281, -0.3211],
                [-0.3211, -0.3211, 0.0000],
                [-0.4281, -0.8562, -0.3211],
            ],
            [
                [3.3157, 1.1923, 1.2391],
                [1.1923, 5.2526, 1.3774],
                [1.2391, 1.3774, 4.2914],
            ],
            [
                [-1.0813, -0.1352, -0.8110],
                [-0.1352, -1.0813, -0.8110],
                [-0.8110, -0.8110, 0.0000],
            ],
            [
                [-2.2845, -0.6527, -0.4895],
                [-0.4895, -0.4895, 0.0000],
                [-0.6527, -1.3054, -0.4895],
            ],
            [
                [-1.0813, -0.1352, -0.8110],
                [-0.1352, -1.0813, -0.8110],
                [-0.8110, -0.8110, 0.0000],
            ],
            [
                [4.2686, 1.5275, 1.4637],
                [1.5275, 6.1970, 1.6014],
                [1.4637, 1.6014, 5.5184],
            ],
        ]
    ),
)


def allclose(a: torch.Tensor, b: torch.Tensor) -> bool:
    assert a.dtype == b.dtype
    atol = 1e-6 if a.dtype == torch.float32 else 1e-12
    return torch.allclose(a, b, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("backend", FastBSR.matmul_backends.keys())
def test_bsr_matmul_small(backend: str, dtype: torch.dtype):
    torch.manual_seed(hash(f"{backend}_{dtype}"))

    matrix_small = matrix.to(device=device, dtype=dtype)
    bsr = FastBSR(matrix_small, backend=backend)

    x = torch.randn(matrix_small.shape[0], device=device, dtype=dtype)
    result = bsr @ x
    expected = matrix_small.to_dense() @ x
    assert allclose(result, expected), "invalid BSR matmul result"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("backend", FastBSR.matmul_backends.keys())
def test_bsr_matmul_large(backend: str, dtype: torch.dtype):
    torch.manual_seed(hash(f"{backend}_{dtype}"))

    block_rows = 1000
    block_cols = 1000
    block_size = 4
    blocks_per_row = 3
    crow_indices = torch.arange(
        0, block_rows * blocks_per_row + 1, blocks_per_row, device=device
    )
    col_indices = torch.randint(
        0, block_cols, (block_rows * blocks_per_row,), device=device
    )
    values = torch.randn(
        (block_rows * blocks_per_row, block_size, block_size),
        device=device,
        dtype=dtype,
    )
    matrix = torch.sparse_bsr_tensor(
        crow_indices=crow_indices,
        col_indices=col_indices,
        values=values,
        size=(block_rows * block_size, block_cols * block_size),
    )

    bsr = FastBSR(matrix, backend=backend)

    x = torch.randn(matrix.shape[0], device=device, dtype=dtype)
    result = bsr @ x
    expected = matrix @ x
    assert allclose(result, expected), "invalid BSR matmul result"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_csr_to_bsr_small(dtype: torch.dtype):
    # 1 2 0 0
    # 0 3 0 0
    # 4 0 5 0
    # 6 7 8 9
    matrix_csr = torch.sparse_csr_tensor(
        crow_indices=torch.tensor([0, 2, 3, 5, 9]),
        col_indices=torch.tensor([0, 1, 1, 0, 2, 0, 1, 2, 3]),
        values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=dtype),
    ).to(device=device, dtype=dtype)

    matrix_bsr = csr_to_bsr(matrix_csr, block_size=2)

    assert matrix_bsr.layout == torch.sparse_bsr
    assert matrix_bsr.dtype == dtype
    assert matrix_bsr.device.type == device.type
    assert matrix_bsr.crow_indices().tolist() == [0, 1, 3]
    assert matrix_bsr.col_indices().tolist() == [0, 0, 1]
    assert allclose(
        matrix_bsr.values(),
        torch.tensor(
            [[[1, 2], [0, 3]], [[4, 0], [6, 7]], [[5, 0], [8, 9]]],
            dtype=dtype,
            device=device,
        ),
    )
    assert matrix_bsr.values().stride() == (4, 1, 2)  # column-major storage


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_csr_to_bsr_medium(dtype: torch.dtype):
    matrix_csr = matrix.to_dense().to_sparse_csr().to(device=device, dtype=dtype)

    matrix_bsr = csr_to_bsr(matrix_csr, block_size=3)
    matrix_expected = matrix.to(device=device, dtype=dtype)
    assert matrix_bsr.layout == torch.sparse_bsr
    assert matrix_bsr.dtype == dtype
    assert matrix_bsr.device.type == device.type
    assert allclose(matrix_bsr.to_dense(), matrix_expected.to_dense())
    assert matrix_bsr.values().stride() == (9, 1, 3)
