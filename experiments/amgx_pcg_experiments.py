import torch
import os
import pandas as pd
from dd_solvers import *
from experiments import *

cg_kwargs = {
    "maxiter": 2000,
    "rtol": 1e-9,
}

solvers = [
    CG(AMGX("L1_TRUNC", torch.float32), **cg_kwargs),
    CG(AMGX("L1_TRUNC"), **cg_kwargs),
    AMGX("CG_L1_TRUNC"),
    CG(AMGX("L1_TRUNC", torch.float32), **cg_kwargs, bsr_matmul=False),
    CG(AMGX("L1_TRUNC"), **cg_kwargs, bsr_matmul=False),
]

results_path = f"../results/experiment_amgx_pcg.csv"

experiments = [
    # (d, max_m, p)
    (2, 11, 1),
    (2, 10, 3),
    (2, 9, 5),
]


def run_experiment(d: int, max_m: int, p: int):
    factory_kwargs = {
        "test_case": continuous_coefficient_2d if d == 2 else continuous_coefficient_3d,
        "polynomial_degree": p,
        "setup_repetitions": 1,
        "solution_warmup_steps": 0,
        "solution_measurement_steps": 1,
        "solution_repetitions": 10,
    }

    mesh_family = UniformMeshes(d=d, m=max_m)
    factory = ExperimentFactory(**factory_kwargs, mesh_family=mesh_family)

    for m in range(1, max_m + 1):
        for solver in solvers:
            factory.add(
                Experiment(
                    coarse_m=None,
                    solvers_m=None,
                    fine_m=f"S{m}",
                    solver=solver,
                )
            )

    return factory.run()


for d, max_m, p in experiments:
    df = run_experiment(d, max_m, p)
    if os.path.exists(results_path):
        df_existing = pd.read_csv(results_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(results_path, index=False)
