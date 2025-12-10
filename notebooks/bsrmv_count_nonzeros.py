import gc
from tqdm import tqdm
import pandas as pd

from experiments import *


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

    nnz = discrete_problem.exact_form_matrix.nnz

    results.append(
        {
            "dimension": dimension_str,
            "mesh m": mesh_n,
            "DOFs": discrete_problem.exact_form_matrix.shape[0],
            "degree": degree,
            "nnz": nnz,
        }
    )

df = pd.DataFrame(results)
print(df)
df.to_csv("bsrmv_nnz.csv", index=False)
