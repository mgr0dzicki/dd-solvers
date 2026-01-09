import pytest
import torch
from dd_solvers.gemv import gemv_strided_batched


@pytest.mark.parametrize(
    "mat_dtype,vec_dtype",
    [
        (torch.float16, torch.float32),
        (torch.float16, torch.float64),
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.float64),
        (torch.float32, torch.float32),
        (torch.float32, torch.float64),
        (torch.float64, torch.float64),
    ],
)
@pytest.mark.parametrize("k", [1, 2, 5, 10, 20, 50, 100, 200])
@pytest.mark.parametrize("use_shared_memory", [True, False])
def test_gemv_strided_batched(mat_dtype: torch.dtype, vec_dtype: torch.dtype, k: int, use_shared_memory: bool):
    n = 7
    torch.manual_seed(42)

    A = (
        torch.randn((n, k, k), device="cuda", dtype=mat_dtype)
        .permute(0, 2, 1)
        .contiguous()
        .permute(0, 2, 1)
    )  # column-major
    b = torch.randn((n, k), device="cuda", dtype=vec_dtype)

    exact = (A.to(b.dtype) @ b[:, :, None]).reshape(n, k)
    result = gemv_strided_batched(A, b, use_shared_memory=use_shared_memory)

    assert torch.allclose(exact, result, rtol=1e-5, atol=1e-5)
