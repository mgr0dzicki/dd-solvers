import sys
from dd_solvers import *
from experiments import *
import torch
import pandas as pd

d = 2
p = 1
coarse_m = 3
solvers_m = 4
fine_m = 5

cg_kwargs = {
    "maxiter": 2000,
    "rtol": 1e-9,
}

factory_kwargs = {
    "test_case": continuous_coefficient_2d if d == 2 else continuous_coefficient_3d,
    "polynomial_degree": p,
}

asm_factory_kwargs = {
    **factory_kwargs,
    "number_of_repetitions": 1,
    "solution_warmup_steps": 10,
    "solution_measurement_steps": 10,
    "solution_repetitions": 10,
}

def run_experiments(mesh_family, ms):
    factory = ExperimentFactory(**factory_kwargs, mesh_family=mesh_family)

    for coarse_m, solvers_m, fine_m in ms:
        factory.add(
            Experiment(
                coarse_m=coarse_m,
                solvers_m=solvers_m,
                fine_m=fine_m,
                solver=CG(ASM(torch.float32, Inv(torch.float16), CUDSS()), **cg_kwargs),
            )
        )
        factory.add(
            Experiment(
                coarse_m=coarse_m,
                solvers_m=solvers_m,
                fine_m=fine_m,
                solver=CG(
                    ASM(
                        torch.float64,
                        Inv(torch.float32),
                        CUDSS(),
                        hybrid=True,
                    ),
                    **cg_kwargs,
                ),
            )
        )
        factory.add(
            Experiment(
                coarse_m=coarse_m,
                solvers_m=solvers_m,
                fine_m=fine_m,
                solver=CG(
                    ASM(
                        torch.float64,
                        Inv(torch.float16),
                        CUDSS(),
                        hybrid=True,
                    ),
                    **cg_kwargs,
                ),
            )
        )

    df1 = factory.run()
    return df1


mesh_family = MeshFamily.simplices_then_cubes(
    d=d,
    fine=fine_m,
    both=coarse_m,
)

df3 = run_experiments(mesh_family, [(coarse_m, solvers_m, fine_m + 1)])
df3.to_csv("single_run_results.csv")
