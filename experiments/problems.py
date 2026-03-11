from typing import Callable, NamedTuple
import numpy as np
from numpy import sin, cos, pi


__all__ = [
    "Problem",
    "Solution",
    "TestCase",
    "constant_coefficient_2d",
    "constant_coefficient_3d",
    "continuous_coefficient_2d",
    "continuous_coefficient_3d",
]


class Problem(NamedTuple):
    delta: float
    rho: Callable[[np.ndarray], np.ndarray]  # (dim, num_points) -> (num_points,)
    f: Callable[[np.ndarray], np.ndarray]  # (dim, num_points) -> (num_points,)


Solution = Callable[[np.ndarray], np.ndarray]


class TestCase(NamedTuple):
    __test__ = False  # to prevent pytest from collecting this class

    name: str
    problem: Problem
    solution: Solution


constant_coefficient_2d = TestCase(
    "constant coefficient 2D",
    Problem(
        delta=7,
        rho=lambda x: np.ones(x.shape[1]),
        f=lambda x: 2 * pi**2 * sin(x[0] * pi) * sin(x[1] * pi),
    ),
    lambda x: sin(x[0] * pi) * sin(x[1] * pi),
)

constant_coefficient_3d = TestCase(
    "constant coefficient 3D",
    Problem(
        delta=17,
        rho=lambda x: np.ones(x.shape[1]),
        f=lambda x: 3 * pi**2 * sin(x[0] * pi) * sin(x[1] * pi) * sin(x[2] * pi),
    ),
    lambda x: sin(x[0] * pi) * sin(x[1] * pi) * sin(x[2] * pi),
)

continuous_coefficient_2d = TestCase(
    "continuous coefficient 2D",
    Problem(
        delta=7,
        rho=lambda x: x[0] ** 2 + x[1] ** 2 + 1,
        f=lambda x: (8 * pi)
        * (
            4
            * pi
            * (x[0] ** 2 + x[1] ** 2 + 1)
            * sin(4 * x[0] * pi)
            * sin(4 * x[1] * pi)
            - x[0] * cos(4 * x[0] * pi) * sin(4 * x[1] * pi)
            - x[1] * sin(4 * x[0] * pi) * cos(4 * x[1] * pi)
        ),
    ),
    lambda x: sin(4 * x[0] * pi) * sin(4 * x[1] * pi),
)

continuous_coefficient_3d = TestCase(
    "continuous coefficient 3D",
    Problem(
        delta=17,
        rho=lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 1,
        f=lambda x: (8 * pi)
        * (
            6
            * pi
            * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 1)
            * sin(4 * x[0] * pi)
            * sin(4 * x[1] * pi)
            * sin(4 * x[2] * pi)
            - x[0] * cos(4 * x[0] * pi) * sin(4 * x[1] * pi) * sin(4 * x[2] * pi)
            - x[1] * sin(4 * x[0] * pi) * cos(4 * x[1] * pi) * sin(4 * x[2] * pi)
            - x[2] * sin(4 * x[0] * pi) * sin(4 * x[1] * pi) * cos(4 * x[2] * pi)
        ),
    ),
    lambda x: sin(4 * x[0] * pi) * sin(4 * x[1] * pi) * sin(4 * x[2] * pi),
)
