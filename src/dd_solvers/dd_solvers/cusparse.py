import torch

torch.ops.dd_solvers_cusparse.init()

__all__ = ["sort_csr", "csr_to_bsr"]


def tensor_size(t: torch.Tensor) -> int:
    return t.element_size() * t.nelement()


def csr_tensor_size(matrix: torch.Tensor) -> int:
    return (
        tensor_size(matrix.crow_indices())
        + tensor_size(matrix.col_indices())
        + tensor_size(matrix.values())
    )


def sort_csr(matrix: torch.Tensor, num_parts: int | None = None) -> torch.Tensor:
    if num_parts is None:
        total_memory = torch.cuda.get_device_properties().total_memory
        allocated_memory = torch.cuda.memory_allocated()
        matrix_size = csr_tensor_size(matrix)
        num_parts = 1 + (6 * matrix_size) // (total_memory - allocated_memory)
        if num_parts > 128:
            raise MemoryError(
                f"Cannot sort CSR matrix: not enough GPU memory (need to split into {num_parts} parts)"
            )

    # Note: it invalides original matrix!

    assert 1 <= num_parts <= matrix.size(0)

    rows, cols = matrix.shape
    crow_indices = matrix.crow_indices().to(torch.int32)
    col_indices = matrix.col_indices().to(torch.int32)
    values = matrix.values()

    all_part_values = []
    row_start = 0
    for part in range(num_parts):
        row_num = rows // num_parts + (1 if part < rows % num_parts else 0)
        row_end = row_start + row_num

        part_col_indices = col_indices[crow_indices[row_start] : crow_indices[row_end]]
        part_values = values[crow_indices[row_start] : crow_indices[row_end]]

        crow_offset = crow_indices[row_start].item()
        crow_indices[row_start : row_end + 1] -= crow_offset
        part_crow_indices = crow_indices[row_start : row_end + 1]

        part_matrix = torch.sparse_csr_tensor(
            part_crow_indices,
            part_col_indices,
            part_values,
            size=(row_end - row_start, cols),
        )
        sorted_part_matrix = torch.ops.dd_solvers_cusparse.sort_csr(part_matrix)
        all_part_values.append(sorted_part_matrix.values())

        crow_indices[row_start : row_end + 1] += crow_offset
        row_start = row_end

    sorted_values = torch.cat(all_part_values, out=values)
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, size=matrix.shape)


def csr_to_bsr(matrix: torch.Tensor, block_size: int) -> torch.Tensor:
    return torch.ops.dd_solvers_cusparse.csr_to_bsr(matrix, block_size)
