# %%
import sys
from dd_solvers import *
from experiments import *
import torch
import pandas as pd

# %%
d = int(sys.argv[1])
p = int(sys.argv[2])
fine_m = int(sys.argv[3])

cases = ["simplicial", "cubical", "mixed"]

# %%
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
    "number_of_repetitions": 3,
    "solution_warmup_steps": 10,
    "solution_measurement_steps": 10,
    "solution_repetitions": 10,
}

# %%
local_solvers_map = {
    "inv": Inv(),
    "inv16": Inv(torch.float16),
    "lu": LU(),
    "cholesky": Cholesky(),
    "cudss": BatchCUDSS(),
}
local_solvers = [Inv(torch.float16)]
print("local solvers: ", local_solvers)

# %%
results_prefix = (
    f"../results/experiment_solvers_new_d{d}_p{p}_f{fine_m}"
)
print("results_prefix: ", results_prefix)


# %%
def run_experiments(mesh_family, ms):
    factory = ExperimentFactory(**factory_kwargs, mesh_family=mesh_family)

    for coarse_m, solvers_m, fine_m in ms:
        for local_solver in local_solvers:
            factory.add(
                Experiment(
                    coarse_m=coarse_m,
                    solvers_m=solvers_m,
                    fine_m=fine_m,
                    solver=CG(ASM(torch.float32, local_solver, CUDSS()), **cg_kwargs),
                )
            )
            # factory.add(
            #     Experiment(
            #         coarse_m=coarse_m,
            #         solvers_m=solvers_m,
            #         fine_m=fine_m,
            #         solver=CG(
            #             ASM(
            #                 torch.float64,
            #                 local_solver,
            #                 CUDSS(),
            #                 hybrid=True,
            #             ),
            #             **cg_kwargs,
            #         ),
            #     )
            # )

    return factory.run()


# %%
if "simplicial" in cases:
    print("Case 1: All meshes simplicial")

    mesh_family = MeshFamily.simplices_then_cubes(
        d=d,
        fine=fine_m,
        both=0,
    )

    ms = [
        (coarse_m, solvers_m, fine_m + 1)
        for coarse_m in range(fine_m - 1, fine_m + 2)
        for solvers_m in range(coarse_m, fine_m + 2)
    ]

    df_simp = run_experiments(mesh_family, ms)
    df_simp["case"] = "simplicial"
    df_simp.to_csv(f"{results_prefix}_simplicial.csv", index=False)

# %%
if "cubical" in cases:
    print("Case 2: coarse and solvers cubical, fine simplicial")

    mesh_family = MeshFamily.simplices_then_cubes(
        d=d,
        fine=fine_m,
        both=fine_m,
    )

    ms = [
        (coarse_m, solvers_m, fine_m + 1)
        for coarse_m in range(fine_m - 2, fine_m + 1)
        for solvers_m in range(coarse_m, fine_m + 1)
    ]

    df_cubical = run_experiments(mesh_family, ms)
    df_cubical["case"] = "cubical"
    df_cubical.to_csv(f"{results_prefix}_cubical.csv", index=False)

# %%
if "mixed" in cases:
    print("Case 3: coarse cubical, solvers and fine simplicial")

    mixed_dfs = []
    for coarse_m in range(fine_m - 2, fine_m + 1):
        print(f"coarse_m: {coarse_m}")
        mesh_family = MeshFamily.simplices_then_cubes(
            d=d,
            fine=fine_m,
            both=coarse_m,
        )
        ms = [
            (coarse_m, solvers_m, fine_m + 1)
            for solvers_m in range(coarse_m + 1, fine_m + 2)
        ]
        df3 = run_experiments(mesh_family, ms)
        df3["case"] = "mixed"
        mixed_dfs.append(df3)
    df_mixed = pd.concat(mixed_dfs, ignore_index=True)
    df_mixed.to_csv(f"{results_prefix}_mixed.csv", index=False)

# %%
