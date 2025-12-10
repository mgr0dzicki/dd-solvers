import sys
from dd_solvers import *
from experiments import *
import torch

d = int(sys.argv[1])
p = int(sys.argv[3]) if len(sys.argv) > 3 else 1
fine_m = int(sys.argv[2])

cg_kwargs = {
    "maxiter": 1200,
    "rtol": 1e-9,
}

factory_kwargs = {
    "test_case": continuous_coefficient_2d if d == 2 else continuous_coefficient_3d,
    "polynomial_degree": p,
}

solvers = []
for amg_config in AMGX.config_names:
    solvers.append(CG(AMGX(amg_config, torch.float32), **cg_kwargs))
    # solvers.append(CG(AMGX(amg_config, torch.float64), **cg_kwargs))
solvers.append(CUDSS())

mesh_family = MeshFamily.simplices_then_cubes(
    d=d,
    fine=fine_m,
    both=0,
)

factory = ExperimentFactory(**factory_kwargs, mesh_family=mesh_family)
for solver in solvers:
    factory.add(
        Experiment(
            fine_m=fine_m + 1,
            solvers_m=None,
            coarse_m=None,
            solver=solver,
        )
    )

df = factory.run()
df.to_csv(
    f"../results/experiment_reference_solvers_d{d}_p{p}_f{fine_m}.csv", index=False
)
