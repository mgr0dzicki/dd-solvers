import pytest

import numpy as np
import torch
from dd_solvers import CUDSS, ASM, Inv, CG

from experiments import *


def test_experiment_factory():
    factory = ExperimentFactory(
        test_case=continuous_coefficient_2d,
        mesh_family=MeshFamily.refinements(
            base=Mesh2D.unit_square_uniform(1, 1),
            n=4,
        ),
        number_of_repetitions=3,
        problem_precision=np.float64,
        error_continuous_norms=(2,),
    )

    factory.add(
        Experiment(
            fine_m=3,
            solvers_m=None,
            coarse_m=None,
            solver=CUDSS(),
        )
    )
    factory.add(
        Experiment(
            fine_m=3,
            solvers_m=2,
            coarse_m=0,
            solver=CG(ASM(torch.float32, Inv(torch.bfloat16), CUDSS())),
        )
    )

    results = factory.run()
    assert len(results) == 6
    assert (results["residual norm"] < 1e-6).all()
    assert (results["error L2 norm"] < 0.5).all()
