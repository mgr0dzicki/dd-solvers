import sys
import os
import pandas as pd
from dd_solvers import *
from experiments import *
import torch

d = int(sys.argv[1])
p = int(sys.argv[2])
fine_m = int(sys.argv[3])
solver_names = (
    sys.argv[4].split(",") if len(sys.argv) > 4 else ["amgx", "amgx64", "cudss"]
)

cg_kwargs = {
    "maxiter": 1200,
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

results_path = "../results/experiment_reference_solvers.csv"

solvers = []

for amg_config in AMGX.preconditioner_config_names:
    if "amgx64" in solver_names:
        solvers.append(CG(AMGX(amg_config), **cg_kwargs))
    if "amgx" in solver_names:
        solvers.append(CG(AMGX(amg_config, torch.float32), **cg_kwargs))

# Should be the last one as it can leave the GPU memory in an inconsistent
# state in case of OOM errors.
if "cudss" in solver_names:
    solvers.append(CUDSS())

if not solvers:
    raise ValueError("No solvers specified!")

mesh_family = UniformMeshes(d=d, m=fine_m)

factory = ExperimentFactory(**factory_kwargs, mesh_family=mesh_family)
for solver in solvers:
    factory.add(
        Experiment(
            fine_m=f"S{fine_m}",
            solvers_m=None,
            coarse_m=None,
            solver=solver,
        )
    )

df = factory.run()

if os.path.exists(results_path):
    df_existing = pd.read_csv(results_path)
    df = pd.concat([df_existing, df], ignore_index=True)
df.to_csv(results_path, index=False)
