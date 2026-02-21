import pytest
import numpy as np
import scipy.sparse as sps

from experiments import Problem, discretize, Mesh2D


def test_discretize_2d():
    delta = 7.0
    problem = Problem(
        delta=delta,
        rho=lambda x: np.ones(x.shape[1]),
        f=lambda x: 2 * np.pi**2 * np.sin(x[0] * np.pi) * np.sin(x[1] * np.pi),
    )
    mesh = Mesh2D.unit_square_uniform(1, 1)
    discrete_problem = discretize(problem, mesh, polynomial_degree=1)

    assert discrete_problem.function_space.ufl_element().degree == 1

    A = discrete_problem.exact_form_matrix.toarray()
    assert A.shape == (6, 6)

    # Elements of the coarse matrix
    assert np.allclose(A[:3, :3].sum(), delta * 5 / 2)
    assert np.allclose(A[3:, :3].sum(), -delta * 1 / 2)
    assert np.allclose(A[3:, 3:].sum(), delta * 5 / 2)
    assert np.allclose(A[:3, 3:].sum(), -delta * 1 / 2)
