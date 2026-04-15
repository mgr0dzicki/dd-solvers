import sys
import os
import pandas as pd
import numpy as np
from dd_solvers import *
from experiments import *
from problems import *
import torch

d = 2
p = 1
fine_m = 5
coefficient_param = 1e6

cg_kwargs = {
    "maxiter": 400,
    "rtol": 1e-6,
    "estimate_cond": True,
}

problem = Problem(
    delta=7,
    rho=(
        lambda x: (
            # coeeficient_param on even and 1 on odd indices
            np.vstack([np.full((x.shape[1] // 2), 1.0), np.random.rand((x.shape[1] // 2)) * coefficient_param]).T.flatten()
        )
    ),
    f=constant_coefficient_2d.problem.f,
)

test_case = TestCase(
    "constant coefficient 2D",
    problem,
    constant_coefficient_2d.solution,
)

factory_kwargs = {
    "test_case": continuous_coefficient_2d if d == 2 else continuous_coefficient_3d,
    "polynomial_degree": p,
    "setup_repetitions": 1,
    "solution_warmup_steps": 0,
    "solution_measurement_steps": 1,
    "solution_repetitions": 1,
}

results_path = "../results/experiment_reference_solvers_madness.csv"

solvers = []

for amg_config in AMGX.preconditioner_config_names:
    solvers.append(CG(AMGX(amg_config), **cg_kwargs))
    solvers.append(CG(AMGX(amg_config, torch.float32), **cg_kwargs))

# Should be the last one as it can leave the GPU memory in an inconsistent
# state in case of OOM errors.
solvers.append(CUDSS())

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
