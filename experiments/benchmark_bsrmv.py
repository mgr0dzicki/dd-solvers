import torch
from tqdm import tqdm
import pandas as pd
import gc
import functools

from experiments import *
import dd_solvers
from benchmark_utils import timeit

meshes_2d = UniformMeshes(d=2, m=11)
meshes_3d = UniformMeshes(d=3, m=7)

problems = [
    ("2D", "S11", 1),
    ("2D", "S10", 2),
    ("2D", "S10", 3),
    ("2D", "S9", 4),
    ("2D", "S8", 5),
    ("3D", "S7", 1),
    ("3D", "S6", 2),
    ("3D", "S5", 3),
]

algorithms = {
    f"bsr_{backend}": functools.partial(
        dd_solvers.FastBSR,
        backend=backend,
    )
    for backend in dd_solvers.FastBSR.matmul_backends.keys()
}
algorithms["csr_cusparse"] = lambda mat: mat

results = []
for problem in tqdm(problems):
    dimension_str, mesh_n, degree = problem
    dimension = int(dimension_str[0])

    if dimension == 2:
        mesh = meshes_2d[mesh_n]
        test_case = continuous_coefficient_2d
    else:
        mesh = meshes_3d[mesh_n]
        test_case = continuous_coefficient_3d

    gc.collect()
    discrete_problem = discretize(
        problem=test_case.problem,
        mesh=mesh,
        polynomial_degree=degree,
    )
    dofs_per_el = discrete_problem.function_space.dofmap.list.shape[1]
    nnz = discrete_problem.exact_form_matrix.nnz
    A = dd_solvers.csr_to_torch(discrete_problem.exact_form_matrix).cuda()
    Ab = dd_solvers.csr_to_bsr(A, block_size=dofs_per_el)
    bsr_nnz = Ab.values().shape[0]

    # To ensure fair comparison
    assert Ab.crow_indices().dtype == torch.int32
    assert Ab.col_indices().dtype == torch.int32
    assert A.crow_indices().dtype == torch.int32
    assert A.col_indices().dtype == torch.int32

    for dtype in [torch.float64]:
        x = torch.randn(
            discrete_problem.exact_form_matrix.shape[0], device="cuda", dtype=dtype
        )
        for algorithm_name, algorithm_constructor in algorithms.items():
            if algorithm_name.startswith("csr"):
                mat = A.to(dtype)
            else:
                mat = Ab.to(dtype)

            gc.collect()
            torch.cuda.empty_cache()

            mat_alg = algorithm_constructor(mat)
            times = [
                timeit(lambda: mat_alg @ x, warmup_iters=5, iters=10, repetitions=5)
                for _ in range(5)
            ]
            error_norm = torch.norm(mat_alg @ x - mat @ x)
            results.append(
                {
                    "dimension": dimension_str,
                    "mesh": mesh_n,
                    "dtype": dtype,
                    "DOFs": discrete_problem.exact_form_matrix.shape[0],
                    "DOFs per element": dofs_per_el,
                    "nnz": nnz,
                    "bsr nnz": bsr_nnz,
                    "degree": degree,
                    "algorithm": algorithm_name,
                    "times [ms]": times,
                    "error norm": error_norm.item(),
                }
            )

df = pd.DataFrame(results)
print(df)
df.to_csv("../results/benchmark_bsrmv.csv", index=False)
