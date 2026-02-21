import sys
from dd_solvers import *
from experiments import *
import torch

d = int(sys.argv[1])
assert d == 2
p = int(sys.argv[2])
n = int(sys.argv[3])

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

ref_solvers = []
for amg_config in AMGX.preconditioner_config_names:
    ref_solvers.append(CG(AMGX(amg_config, torch.float32), **cg_kwargs))

schwarz_solvers = [
    CG(AdditiveSchwarz(torch.float32, Inv(torch.float16), CUDSS()), **cg_kwargs),
    CG(HybridSchwarz(torch.float64, Inv(torch.float16), CUDSS()), **cg_kwargs),
]

results_path = f"../results/experiment_solvers_unstructured_d{d}_p{p}_n{n}.csv"
print("results path: ", results_path)

print("Generating mesh family...")
mesh_family = UnstructuredMeshes(n=n)
factory = ExperimentFactory(**factory_kwargs, mesh_family=mesh_family)

ms = [("C", "C", "F"), ("C", "F", "F"), ("F", "F", "F")]

for solver in ref_solvers:
    factory.add(
        Experiment(
            coarse_m=None,
            solvers_m=None,
            fine_m="F",
            solver=solver,
        )
    )

for coarse_m, solvers_m, fine_m_str in ms:
    for solver in schwarz_solvers:
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
