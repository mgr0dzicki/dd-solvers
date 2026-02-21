from dd_solvers import *
from experiments import *
import torch
import pandas as pd

cg_kwargs = {
    "maxiter": 2000,
    "rtol": 1e-9,
}

factory_kwargs = {
    "setup_repetitions": 1,
    "solution_warmup_steps": 0,
    "solution_measurement_steps": 1,
    "solution_repetitions": 10,
}

preconditioner_factory_kwargs = {
    "setup_repetitions": 1,
    "solution_warmup_steps": 0,
    "solution_warmup_steps": 10,
    "solution_measurement_steps": 10,
}

preconditioner_args_combinations = [
    (torch.float64, Inv(), CUDSS()),
    (torch.float64, Inv(torch.float32), CUDSS()),
    (torch.float64, Inv(torch.float16), CUDSS()),
    (torch.float64, Inv(torch.bfloat16), CUDSS()),
    (torch.float32, Inv(), CUDSS()),
    (torch.float32, Inv(torch.float16), CUDSS()),
    (torch.float32, Inv(torch.bfloat16), CUDSS()),
]

experiments = [
    # (d, p, fine_m)
    (2, 1, 10),
    (2, 1, 11),
    (2, 3, 9),
    (2, 5, 8),
]

results_path = "../results/experiment_precisions.csv"


def ms(fine_m: int):
    ms_simp = [
        (f"S{coarse_m}", f"S{solvers_m}", f"S{fine_m}")
        for coarse_m in range(fine_m - 2, fine_m + 1)
        for solvers_m in (coarse_m,)
    ]
    ms_cub = []
    ms_mixed = []

    return ms_simp + ms_cub + ms_mixed


def run_experiments(d: int, p: int, fine_m: int):
    mesh_family = UniformMeshes(d=d, m=fine_m)

    test_case_kwargs = {
        "test_case": continuous_coefficient_2d if d == 2 else continuous_coefficient_3d,
        "polynomial_degree": p,
        "mesh_family": mesh_family,
    }

    factory = ExperimentFactory(**test_case_kwargs, **factory_kwargs)
    factory_preconditioner = ExperimentFactory(
        **test_case_kwargs, **preconditioner_factory_kwargs
    )

    for coarse_m, solvers_m, fine_m in ms(fine_m):
        for schwarz_args in preconditioner_args_combinations:
            for preconditioner in (
                AdditiveSchwarz,
                HybridSchwarz,
            ):
                factory.add(
                    Experiment(
                        coarse_m=coarse_m,
                        solvers_m=solvers_m,
                        fine_m=fine_m,
                        solver=CG(preconditioner(*schwarz_args), **cg_kwargs),
                    )
                )
                factory_preconditioner.add(
                    Experiment(
                        coarse_m=coarse_m,
                        solvers_m=solvers_m,
                        fine_m=fine_m,
                        solver=preconditioner(*schwarz_args, collect_timings=True),
                    )
                )

    df1 = factory.run()
    df2 = factory_preconditioner.run()
    df = pd.concat([df1, df2], ignore_index=True)
    return df


df = pd.DataFrame()
for d, p, fine_m in experiments:
    print(f"Running experiments for d={d}, p={p}, fine_m={fine_m}...")
    local_df = run_experiments(d, p, fine_m)
    df = pd.concat([df, local_df], ignore_index=True)
    df.to_csv(results_path, index=False)  # Save intermediate results
