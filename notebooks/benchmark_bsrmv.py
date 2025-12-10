import torch
from tqdm import tqdm
import pandas as pd
import gc
import functools

from experiments import *
import dd_solvers


def time_mul(mat, x):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(20):
        _ = mat @ x

    start.record()
    for _ in range(100):
        _ = mat @ x
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / 100.0


meshes_2d = MeshFamily.refinements(
    base=Mesh2D.unit_square_uniform(1, 1),
    n=12,
)
meshes_3d = MeshFamily.refinements(
    base=Mesh3D.unit_cube_uniform(1, 1, 1),
    n=8,
)

problems = [
    ("2D", 11, 1),
    ("2D", 10, 1),
    ("2D", 10, 2),
    ("2D", 10, 3),
    ("2D", 9, 4),
    ("2D", 8, 5),
    ("3D", 7, 1),
    ("3D", 6, 1),
    ("3D", 6, 2),
    ("3D", 5, 2),
    ("3D", 5, 3),
]

algorithms = {
    f"bsr_{backend}": functools.partial(dd_solvers.FastBSR, backend=backend)
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
    Abs = discrete_problem.exact_form_matrix.tobsr((dofs_per_el, dofs_per_el))
    Ab = torch.sparse_bsr_tensor(
        torch.as_tensor(Abs.indptr),
        torch.as_tensor(Abs.indices),
        torch.as_tensor(Abs.data),
        Abs.shape,
    ).cuda()
    A = dd_solvers.csr_to_torch(discrete_problem.exact_form_matrix).cuda()

    # To ensure fair comparison
    assert Ab.crow_indices().dtype == torch.int32
    assert Ab.col_indices().dtype == torch.int32
    assert A.crow_indices().dtype == torch.int32
    assert A.col_indices().dtype == torch.int32

    for dtype in [torch.float32, torch.float64]:
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
            time = time_mul(mat_alg, x)
            error_norm = torch.norm(mat_alg @ x - mat @ x)
            results.append(
                {
                    "dimension": dimension_str,
                    "mesh m": mesh_n,
                    "dtype": dtype,
                    "DOFs": discrete_problem.exact_form_matrix.shape[0],
                    "DOFs per element": dofs_per_el,
                    "degree": degree,
                    "algorithm": algorithm_name,
                    "time [ms]": time,
                    "error norm": error_norm.item(),
                }
            )

df = pd.DataFrame(results)
print(df)
df.to_csv("benchmark_bsrmv.csv", index=False)
