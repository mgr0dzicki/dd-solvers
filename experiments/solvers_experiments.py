import sys
import os
from dd_solvers import *
from experiments import *
import torch
import pandas as pd

d = int(sys.argv[1])
p = int(sys.argv[2])
fine_m = int(sys.argv[3])

cg_kwargs = {
    "maxiter": 2000,
    "rtol": 1e-9,
}

factory_kwargs = {
    "test_case": continuous_coefficient_2d if d == 2 else continuous_coefficient_3d,
    "polynomial_degree": p,
    "setup_repetitions": 1,
    "solution_warmup_steps": 0,
    "solution_measurement_steps": 1,
    "solution_repetitions": 10,
}

solvers = [
    CG(AdditiveSchwarz(torch.float32, Inv(torch.float16), CUDSS()), **cg_kwargs),
    CG(HybridSchwarz(torch.float64, Inv(torch.float16), CUDSS()), **cg_kwargs),
]

results_path = f"../results/experiment_solvers.csv"

mesh_family = UniformMeshes(d=d, m=fine_m)
factory = ExperimentFactory(**factory_kwargs, mesh_family=mesh_family)

ms_simp = [
    (f"S{coarse_m}", f"S{solvers_m}", f"S{fine_m}")
    for coarse_m in range(fine_m - 2, fine_m + 1)
    for solvers_m in range(coarse_m, fine_m + 1)
]
ms_cub = [
    (f"C{coarse_m}", f"C{solvers_m}", f"S{fine_m}")
    for coarse_m in range(fine_m - 2, fine_m + 1)
    for solvers_m in range(coarse_m, fine_m + 1)
]
ms_mixed = [
    (f"C{coarse_m}", f"S{solvers_m}", f"S{fine_m}")
    for coarse_m in range(fine_m - 2, fine_m + 1)
    for solvers_m in range(coarse_m, fine_m + 1)
]

ms = ms_simp + ms_cub + ms_mixed

for coarse_m, solvers_m, fine_m_str in ms:
    for solver in solvers:
        factory.add(
            Experiment(
                coarse_m=coarse_m,
                solvers_m=solvers_m,
                fine_m=fine_m_str,
                solver=solver,
            )
        )

df = factory.run()

if os.path.exists(results_path):
    df_existing = pd.read_csv(results_path)
    df = pd.concat([df_existing, df], ignore_index=True)
df.to_csv(results_path, index=False)
