from typing import Callable
import torch
import math
from numba import cuda

__all__ = ["FastBSR"]

type MatmulBackend = Callable[[torch.Tensor], torch.Tensor]


def ceildiv(a: int, b: int):
    return (a + b - 1) // b


class FastBSR:
    def __init__(
        self, matrix: torch.Tensor, backend: str = "row_per_thread_irrespective"
    ):
        assert matrix.is_cuda
        assert matrix.layout is torch.sparse_bsr

        self._matmul_backend = self.matmul_backends[backend](matrix)

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        return self._matmul_backend(x)

    @staticmethod
    def _matmul_cusparse(matrix: torch.Tensor) -> MatmulBackend:
        def matmul(x: torch.Tensor) -> torch.Tensor:
            return matrix @ x

        return matmul

    @staticmethod
    def _kernel_signature(matrix: torch.Tensor) -> str:
        it = str(matrix.col_indices().dtype).split(".")[-1]
        vt = str(matrix.values().dtype).split(".")[-1]
        return f"void({it}[:], {it}[:], {vt}[:, :, :], {vt}[:], {vt}[:])"

    @staticmethod
    def _matmul_row_per_thread_irrespective(
        matrix: torch.Tensor, thread_block_size: int = 256
    ) -> MatmulBackend:
        bs = matrix.values().shape[1]

        @cuda.jit(FastBSR._kernel_signature(matrix))
        def bsrmv_row_per_thread_irrespective_kernel(
            crow_indices, col_indices, values, x, output
        ):
            target_row = cuda.grid(1)
            target_block_row = target_row // bs
            if target_block_row >= crow_indices.size - 1:
                return

            r = target_row % bs
            first_block = crow_indices[target_block_row]
            last_block = crow_indices[target_block_row + 1]

            local_out = 0.0
            for block in range(first_block, last_block):
                col_offset = col_indices[block] * bs
                for c in range(bs):
                    A_elem = values[block, r, c]
                    x_elem = x[col_offset + c]
                    local_out += A_elem * x_elem
            output[target_row] = local_out

        def bsrmv_row_per_thread_irrespective(x):
            output = torch.empty_like(x)
            grid_size = ceildiv(matrix.shape[0], thread_block_size)
            bsrmv_row_per_thread_irrespective_kernel[
                (grid_size,), (thread_block_size,)
            ](matrix.crow_indices(), matrix.col_indices(), matrix.values(), x, output)
            return output

        return bsrmv_row_per_thread_irrespective

    @staticmethod
    def _matmul_column_by_column(
        matrix: torch.Tensor, thread_block_size: int = 256
    ) -> MatmulBackend:
        bs = matrix.values().shape[1]
        initial_stride = 1 << int(math.ceil(math.log2((32 / bs) / 2)))

        @cuda.jit(FastBSR._kernel_signature(matrix))
        def bsrmv_column_by_column_kernel(crow_indices, col_indices, values, x, output):
            target_block_row = cuda.grid(1) // 32
            lane = cuda.threadIdx.x % 32
            first_block = crow_indices[target_block_row]
            last_block = crow_indices[target_block_row + 1]
            target_col = first_block * bs + lane // bs
            r = lane % bs
            sync_mask = cuda.ballot_sync(0xFFFFFFFF, lane < (32 // bs) * bs)
            if lane < (32 // bs) * bs:
                local_out = 0.0
                while target_col < last_block * bs:
                    block = target_col // bs
                    c = target_col % bs
                    A_elem = values[block, r, c]
                    x_elem = x[col_indices[block] * bs + c]
                    local_out += x_elem * A_elem
                    target_col += 32 // bs

                stride = initial_stride
                while stride >= 1:
                    other = cuda.shfl_down_sync(sync_mask, local_out, stride * bs)
                    if lane < stride * bs and lane + stride * bs < 32:
                        local_out += other
                    stride >>= 1
                if lane < bs:
                    output[target_block_row * bs + lane] = local_out

        def bsrmv_column_by_column(x: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            block_rows = matrix.crow_indices().size(0) - 1
            block_rows_per_thread_block = thread_block_size // 32
            grid_size = ceildiv(block_rows, block_rows_per_thread_block)
            bsrmv_column_by_column_kernel[(grid_size,), (thread_block_size,)](
                matrix.crow_indices(), matrix.col_indices(), matrix.values(), x, output
            )
            return output

        return bsrmv_column_by_column

    @staticmethod
    def _matmul_block_by_block(
        matrix: torch.Tensor, thread_block_size: int = 256
    ) -> MatmulBackend:
        bs = matrix.values().shape[1]
        initial_stride = 1 << int(math.ceil(math.log2((32 / bs) / 2)))

        @cuda.jit(FastBSR._kernel_signature(matrix))
        def bsrmv_block_by_block_kernel(crow_indices, col_indices, values, x, output):
            target_block_row = cuda.grid(1) // 32
            lane = cuda.threadIdx.x % 32
            first_block = crow_indices[target_block_row]
            last_block = crow_indices[target_block_row + 1]
            target_block = first_block + lane // (bs * bs)
            c = (lane // bs) % bs
            r = lane % bs
            sync_mask = cuda.ballot_sync(0xFFFFFFFF, lane < (32 // bs // bs) * bs * bs)
            if lane < (32 // bs // bs) * bs * bs:
                local_out = 0.0
                while target_block < last_block:
                    block = target_block
                    A_elem = values[block, r, c]
                    x_elem = x[col_indices[block] * bs + c]
                    local_out += x_elem * A_elem
                    target_block += 32 // (bs * bs)

                stride = initial_stride
                while stride >= 1:
                    other = cuda.shfl_down_sync(sync_mask, local_out, stride * bs)
                    if lane < stride * bs and lane + stride * bs < 32:
                        local_out += other
                    stride >>= 1
                if lane < bs:
                    output[target_block_row * bs + lane] = local_out

        def bsrmv_block_by_block(x: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            block_rows = matrix.crow_indices().size(0) - 1
            block_rows_per_thread_block = thread_block_size // 32
            grid_size = ceildiv(block_rows, block_rows_per_thread_block)
            bsrmv_block_by_block_kernel[(grid_size,), (thread_block_size,)](
                matrix.crow_indices(), matrix.col_indices(), matrix.values(), x, output
            )
            return output

        return bsrmv_block_by_block

    @staticmethod
    def _matmul_row_per_thread(matrix: torch.Tensor):
        bs = matrix.values().shape[1]

        @cuda.jit(FastBSR._kernel_signature(matrix))
        def bsrmv_row_per_thread_kernel(crow_indices, col_indices, values, x, output):
            target_block_row = cuda.blockIdx.x
            r = cuda.threadIdx.x
            first_block = crow_indices[target_block_row]
            last_block = crow_indices[target_block_row + 1]
            if r < bs:
                local_out = 0
                for block in range(first_block, last_block):
                    for c in range(bs):
                        A_elem = values[block, r, c]
                        x_elem = x[col_indices[block] * bs + c]
                        local_out += x_elem * A_elem
                output[target_block_row * bs + r] = local_out

        def bsrmv_row_per_thread(x):
            output = torch.empty_like(x)
            grid_size = matrix.crow_indices().size(0) - 1
            thread_block_size = ceildiv(bs, 32) * 32
            bsrmv_row_per_thread_kernel[(grid_size,), (thread_block_size,)](
                matrix.crow_indices(), matrix.col_indices(), matrix.values(), x, output
            )
            return output

        return bsrmv_row_per_thread

    @staticmethod
    def _matmul_row_per_thread_sync(matrix: torch.Tensor):
        bs = matrix.values().shape[1]

        @cuda.jit(FastBSR._kernel_signature(matrix))
        def bsrmv_row_per_thread_sync_kernel(
            crow_indices, col_indices, values, x, output
        ):
            target_block_row = cuda.blockIdx.x
            r = cuda.threadIdx.x
            first_block = crow_indices[target_block_row]
            last_block = crow_indices[target_block_row + 1]
            shared_x = cuda.shared.array((bs,), dtype=values.dtype)
            if r < bs:
                local_out = 0
                for block in range(first_block, last_block):
                    cuda.syncthreads()
                    shared_x[r] = x[col_indices[block] * bs + r]
                    cuda.syncthreads()
                    for c in range(bs):
                        A_elem = values[block, r, c]
                        local_out += shared_x[c] * A_elem
                output[target_block_row * bs + r] = local_out

        def bsrmv_row_per_thread_sync(x):
            output = torch.empty_like(x)
            grid_size = matrix.crow_indices().size(0) - 1
            thread_block_size = ceildiv(bs, 32) * 32
            bsrmv_row_per_thread_sync_kernel[(grid_size,), (thread_block_size,)](
                matrix.crow_indices(), matrix.col_indices(), matrix.values(), x, output
            )
            return output

        return bsrmv_row_per_thread_sync

    matmul_backends = {
        "cusparse": _matmul_cusparse,
        "row_per_thread_irrespective": _matmul_row_per_thread_irrespective,
        "column_by_column": _matmul_column_by_column,
        "block_by_block": _matmul_block_by_block,
        "row_per_thread": _matmul_row_per_thread,
        "row_per_thread_sync": _matmul_row_per_thread_sync,
    }
