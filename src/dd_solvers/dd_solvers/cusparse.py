import torch
import numba

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

from numba import cuda

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

    torch.cat(all_part_values, out=values)
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, size=matrix.shape)


def csr_to_bsr(matrix: torch.Tensor, block_size: int) -> torch.Tensor:
    return torch.ops.dd_solvers_cusparse.csr_to_bsr(matrix, block_size)


@cuda.jit
def gather_csr_count_kernel(crow_indices, col_indices, output):
    row = cuda.grid(1)
    if row >= crow_indices.size - 1:
        return

    count = 0
    prev = -1
    for ind in range(crow_indices[row], crow_indices[row + 1]):
        col = col_indices[ind]
        if col != prev:
            count += 1
            prev = col

    output[row] = count


@cuda.jit
def gather_csr_sum_kernel(
    crow_indices, crow_indices_out, col_indices, col_indices_out, values, values_out
):
    row = cuda.grid(1)
    if row >= crow_indices.size - 1:
        return

    out_pos = crow_indices_out[row]
    prev = -1
    sum = 0
    for ind in range(crow_indices[row], crow_indices[row + 1]):
        col = col_indices[ind]
        val = values[ind]
        if col == prev:
            sum += val
        else:
            if prev != -1:
                col_indices_out[out_pos] = prev
                values_out[out_pos] = sum
                out_pos += 1
            prev = col
            sum = val

    if prev != -1:
        col_indices_out[out_pos] = prev
        values_out[out_pos] = sum


def gather_csr(matrix: torch.Tensor) -> torch.Tensor:
    crow_indices_out = torch.empty_like(matrix.crow_indices())

    thread_block_size = 32
    grid_size = (matrix.shape[0] + thread_block_size - 1) // thread_block_size
    gather_csr_count_kernel[grid_size, thread_block_size](
        matrix.crow_indices(),
        matrix.col_indices(),
        crow_indices_out[1:],
    )
    crow_indices_out[0] = 0
    crow_indices_out[1:].cumsum_(dim=0)
    nnz = crow_indices_out[-1].item()

    col_indices_out = torch.empty(
        (nnz,), dtype=matrix.col_indices().dtype, device=matrix.col_indices().device
    )
    values_out = torch.empty(
        (nnz,), dtype=matrix.values().dtype, device=matrix.values().device
    )
    gather_csr_sum_kernel[grid_size, thread_block_size](
        matrix.crow_indices(),
        crow_indices_out,
        matrix.col_indices(),
        col_indices_out,
        matrix.values(),
        values_out,
    )

    return torch.sparse_csr_tensor(
        crow_indices_out,
        col_indices_out,
        values_out,
        size=matrix.shape,
    )
