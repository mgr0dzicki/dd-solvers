from typing import Callable
import torch
import math
from numba import cuda, types
import cupy as cp
import re
import ctypes
import numpy as np

__all__ = ["FastBSR"]

type MatmulBackend = Callable[[torch.Tensor], torch.Tensor]


def ceildiv(a: int, b: int):
    return (a + b - 1) // b


class FastBSR:
    def __init__(self, matrix: torch.Tensor, backend: str | None = None):
        assert matrix.is_cuda
        assert matrix.layout is torch.sparse_bsr

        if backend is None:
            bs = matrix.values().shape[1]
            backend = (
                "row_per_thread_irrespective" if bs < 15 else "row_per_thread_sync"
            )

        self._matmul_backend = self.matmul_backends[backend](matrix)

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        return self._matmul_backend(x)

    @staticmethod
    def _matmul_cusparse(matrix: torch.Tensor) -> MatmulBackend:
        def matmul(x: torch.Tensor) -> torch.Tensor:
            return matrix @ x

        return matmul

    @staticmethod
    def _matmul_row_per_thread_irrespective(
        matrix: torch.Tensor, thread_block_size: int = 256
    ) -> MatmulBackend:
        bs = matrix.values().shape[1]

        # Define types for pointers
        f64_ptr = types.CPointer(types.float64)
        i32_ptr = types.CPointer(types.int32)

        # 1. Rewrite kernel to accept raw pointers and explicit sizes
        # Signature: (crow_ptr, col_ptr, val_ptr, x_ptr, out_ptr, num_rows, bs)
        @cuda.jit(types.void(i32_ptr, i32_ptr, f64_ptr, f64_ptr, f64_ptr, types.int32, types.int32))
        def bsrmv_raw(crow_p, col_p, val_p, x_p, out_p, num_rows, bs):
            target_row = cuda.grid(1)
            target_block_row = target_row // bs

            if target_block_row >= num_rows:
                return

            # In Numba, pointers support array-like indexing: p[i]
            # Note: We must manage 3D indexing manually since we have a flat pointer
            r = target_row % bs
            first_block = crow_p[target_block_row]
            last_block = crow_p[target_block_row + 1]

            local_out = 0.0
            for block in range(first_block, last_block):
                col_offset = col_p[block] * bs
                for c in range(bs):
                    # Manual 3D index: [block, r, c] -> flat index
                    # Assuming values is (num_blocks, bs, bs) C-contiguous
                    val_idx = (block * bs * bs) + (r * bs) + c

                    A_elem = val_p[val_idx]
                    x_elem = x_p[col_offset + c]
                    local_out += A_elem * x_elem

            out_p[target_row] = local_out

        # 2. Get PTX
        sig = (i32_ptr, i32_ptr, f64_ptr, f64_ptr, f64_ptr, types.int32, types.int32)
        ptx_code = bsrmv_raw.inspect_asm(sig)
        # Find the mangled kernel name using Regex
        # Looks for: .visible .entry _Z12kernel_name... (
        match = re.search(r"\.visible\s+\.entry\s+(\S+)\s*\(", ptx_code)

        if not match:
            raise RuntimeError("Could not find kernel entry point in PTX code.")

        kernel_name = match.group(1)
        print(f"Found kernel name: {kernel_name}")

        # 3. Load using Low-Level Module API (This accepts bytes!)
        module = cp.cuda.function.Module()
        module.load(ptx_code.encode('utf-8'))
        raw_func = module.get_function(kernel_name)

        def bsrmv_row_per_thread_irrespective(x):
            output = torch.empty_like(x)

            # Calculate grid
            num_rows = matrix.shape[0] # assuming matrix is available in scope or passed
            bs = matrix.values().shape[1]
            grid_size = (num_rows + 256 - 1) // 256

            # 4. Launch passing raw memory addresses (integers)
            # Args: (crow_ptr, col_ptr, val_ptr, x_ptr, out_ptr, num_rows, bs)
            # Note: arg_list must be a tuple
            args = (
                matrix.crow_indices().data_ptr(),
                matrix.col_indices().data_ptr(),
                matrix.values().data_ptr(),
                x.data_ptr(),
                output.data_ptr(),
                np.int32(matrix.crow_indices().size(0) - 1), # num_rows (blocks)
                np.int32(bs)
            )

            # raw_func(grid, block, args_tuple)
            raw_func((grid_size,), (256,), args)

            return output

        return bsrmv_row_per_thread_irrespective

    @staticmethod
    def _matmul_column_by_column(
        matrix: torch.Tensor, thread_block_size: int = 256
    ) -> MatmulBackend:
        bs = matrix.values().shape[1]
        initial_stride = 1 << int(math.ceil(math.log2((32 / bs) / 2)))

        @cuda.jit
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

        @cuda.jit
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

        @cuda.jit
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

        @cuda.jit
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
