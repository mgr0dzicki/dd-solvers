from dd_solvers import *
from experiments import *
import torch

fine_m = 11
mesh_family = UniformMeshes(d=2, m=fine_m)

common_kwargs = {
    "test_case": continuous_coefficient_2d,
    "polynomial_degree": 1,
}

preconditioner_factory_kwargs = {
    **common_kwargs,
    "setup_repetitions": 1,
    "solution_warmup_steps": 1,
    "solution_measurement_steps": 1,
    "solution_repetitions": 1,
}

preconditioner_factory = ExperimentFactory(
    **preconditioner_factory_kwargs, mesh_family=mesh_family
)

ms = [
    (f"C{solvers_m}", f"C{solvers_m}", f"S{fine_m}")
    for solvers_m in range(2, fine_m + 1)
] + [
    (f"S{solvers_m}", f"S{solvers_m}", f"S{fine_m}")
    for solvers_m in range(2, fine_m + 1)
]

preconditioners = [
    AdditiveSchwarz(prec, local_solver, CUDSS(), collect_timings=True)
    for prec in [torch.float32, torch.float64]
    for local_solver in [BatchCUDSS(), Inv(), Inv(torch.float16)]
]

for coarse_m, solvers_m, fine_m_str in ms:
    for preconditioner in preconditioners:
        preconditioner_factory.add(
            Experiment(
                coarse_m=coarse_m,
                solvers_m=solvers_m,
                fine_m=fine_m_str,
                solver=preconditioner,
            )
        )


df = preconditioner_factory.run()

df.to_csv("../results/compare_local_solvers.csv")
