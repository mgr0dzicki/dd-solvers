import torch


@torch.library.custom_op("dd_solvers::gemv_strided_batched", mutates_args=())
def gemv_strided_batched(
    matrices: torch.Tensor, vectors: torch.Tensor, use_shared_memory: bool | None = None
) -> torch.Tensor:
    if use_shared_memory is None:
        if vectors.dtype is torch.float32:
            use_shared_memory = vectors.shape[1] > 15 and vectors.shape[1] <= 1024
        elif vectors.dtype is torch.float64:
            use_shared_memory = vectors.shape[1] > 5 and vectors.shape[1] <= 1024
        else:
            raise ValueError("Unsupported data type")

    return torch.ops.dd_solvers_gemv.gemv_strided_batched(
        matrices, vectors, use_shared_memory
    )


@gemv_strided_batched.register_fake
def gemv_strided_batched_fake(
    matrices: torch.Tensor, vectors: torch.Tensor, use_shared_memory: bool = False
) -> torch.Tensor:
    if matrices.ndim != 3 or vectors.ndim != 2:
        raise RuntimeError("matrices must be 3D and vectors must be 2D")
    if matrices.shape[0] != vectors.shape[0]:
        raise RuntimeError("matrices and vectors must have the same batch size")

    return torch.empty_like(vectors)
