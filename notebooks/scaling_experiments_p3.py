import sys
from dd_solvers import *
from experiments import *
import torch

d = 2
p = 3

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

asm_factory_kwargs = {
    **factory_kwargs,
    "setup_repetitions": 3,
    "solution_warmup_steps": 10,
    "solution_measurement_steps": 10,
    "solution_repetitions": 10,
}

solvers = [
    CG(AdditiveSchwarz(torch.float32, Inv(torch.float16), CUDSS()), **cg_kwargs),
    CG(HybridSchwarz(torch.float64, Inv(torch.float16), CUDSS()), **cg_kwargs),
    CG(AMGX("L1_TRUNC", torch.float32), **cg_kwargs),
    CG(AMGX("L1_TRUNC"), **cg_kwargs),
    AMGX("CG_L1_TRUNC"),
    CG(AdditiveSchwarz(torch.float32, Inv(torch.float16), CUDSS()), **cg_kwargs, bsr_matmul=False),
    CG(HybridSchwarz(torch.float64, Inv(torch.float16), CUDSS()), **cg_kwargs, bsr_matmul=False),
    CG(AMGX("L1_TRUNC", torch.float32), **cg_kwargs, bsr_matmul=False),
    CG(AMGX("L1_TRUNC"), **cg_kwargs, bsr_matmul=False),
]

results_path = f"../results/experiment_scaling_d{d}_p{p}.csv"
print("results path: ", results_path)

print("Generating mesh family...")
mesh_family = UniformMeshes(d=d, m=9)
factory = ExperimentFactory(**factory_kwargs, mesh_family=mesh_family)

# ms = [(f"C{m}", f"C{m}", f"S{m}") for m in range(4, 9 + 1)]
ms = [(f"C{m}", f"C{m}", f"S{m}") for m in range(4, 9 + 1)]

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
df.to_csv(results_path, index=False)
