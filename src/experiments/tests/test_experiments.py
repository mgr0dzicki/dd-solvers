import pytest

import numpy as np
import torch
from dd_solvers import CUDSS, AdditiveSchwarz, Inv, CG

from experiments import *


def test_experiment_factory():
    factory = ExperimentFactory(
        test_case=continuous_coefficient_2d,
        mesh_family=UniformMeshes(
            d=2,
            m=4,
        ),
        problem_precision=np.float64,
        error_continuous_norms=(2,),
    )

    factory.add(
        Experiment(
            fine_m="S3",
            solvers_m=None,
            coarse_m=None,
            solver=CUDSS(),
        )
    )
    factory.add(
        Experiment(
            fine_m="S3",
            solvers_m="C3",
            coarse_m="C0",
            solver=CG(AdditiveSchwarz(torch.float32, Inv(torch.float16), CUDSS())),
        )
    )

    results = factory.run()
    assert len(results) == 2
    assert results["exception"].isnull().all()
    assert (results["residual norm"] < 1e-6).all()
    assert (results["error L2 norm"] < 0.5).all()
