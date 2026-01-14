import sys
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

common_kwargs = {
    "test_case": continuous_coefficient_2d if d == 2 else continuous_coefficient_3d,
    "polynomial_degree": p,
}

factory_kwargs = {
    **common_kwargs,
    "setup_repetitions": 1,
    "solution_warmup_steps": 0,
    "solution_measurement_steps": 1,
    "solution_repetitions": 10,
}

preconditioner_factory_kwargs = {
    **common_kwargs,
    "setup_repetitions": 1,
    "solution_warmup_steps": 10,
    "solution_measurement_steps": 10,
    "solution_repetitions": 10,
}


def preconditioners(**kwargs):
    return [
        AdditiveSchwarz(torch.float32, Inv(torch.float16), CUDSS(), **kwargs),
        HybridSchwarz(torch.float64, Inv(torch.float16), CUDSS(), **kwargs),
    ]


results_path = f"../results/experiment_parallelism_d{d}_p{p}_f{fine_m}.csv"
print("results path: ", results_path)

print("Generating mesh family...")
mesh_family = UniformMeshes(d=d, m=fine_m)

factory = ExperimentFactory(**factory_kwargs, mesh_family=mesh_family)
preconditioner_factory = ExperimentFactory(
    **preconditioner_factory_kwargs, mesh_family=mesh_family
)

ms_simp = [
    (f"S{coarse_m}", f"S{solvers_m}", f"S{fine_m}")
    for coarse_m in range(fine_m - 3, fine_m + 1)
    for solvers_m in range(coarse_m, fine_m + 1)
]
ms_cub = [
    (f"C{coarse_m}", f"C{solvers_m}", f"S{fine_m}")
    for coarse_m in range(fine_m - 3, fine_m + 1)
    for solvers_m in range(coarse_m, fine_m + 1)
]
ms_mixed = [
    (f"C{coarse_m}", f"S{solvers_m}", f"S{fine_m}")
    for coarse_m in range(fine_m - 3, fine_m + 1)
    for solvers_m in range(coarse_m, fine_m + 1)
]

ms = ms_simp + ms_cub + ms_mixed

for coarse_m, solvers_m, fine_m_str in ms:
    for preconditioner in preconditioners():
        factory.add(
            Experiment(
                coarse_m=coarse_m,
                solvers_m=solvers_m,
                fine_m=fine_m_str,
                solver=CG(preconditioner, **cg_kwargs),
            )
        )
    for preconditioner in preconditioners(collect_timings=True):
        preconditioner_factory.add(
            Experiment(
                coarse_m=coarse_m,
                solvers_m=solvers_m,
                fine_m=fine_m_str,
                solver=preconditioner,
            )
        )


df = factory.run()
df.to_csv(results_path, index=False)

df2 = preconditioner_factory.run()
df_full = pd.concat([df, df2])
df_full.to_csv(results_path, index=False)
