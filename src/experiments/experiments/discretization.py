from typing import NamedTuple
from dolfinx import fem
import ufl
import numpy as np
import scipy.sparse as sps

from .meshing import Mesh
from .problems import Problem

__all__ = ["DiscreteProblem", "discretize"]


class DiscreteProblem(NamedTuple):
    function_space: fem.FunctionSpace
    exact_form_matrix: sps.csr_matrix
    approximate_form_matrix: sps.csr_matrix | None
    load_vector: np.ndarray

    def astype(self, t: np.dtype):
        return DiscreteProblem(
            function_space=self.function_space,
            exact_form_matrix=self.exact_form_matrix.astype(t),
            approximate_form_matrix=(
                self.approximate_form_matrix.astype(t)
                if self.approximate_form_matrix
                else None
            ),
            load_vector=self.load_vector.astype(t),
        )


def discretize(
    problem: Problem,
    mesh: Mesh,
    polynomial_degree: int = 1,
    assemble_approximate_form: bool = False,
) -> DiscreteProblem:
    V = fem.functionspace(mesh.dolfinx_mesh, ("DG", polynomial_degree))
    delta_p2 = problem.delta * polynomial_degree**2

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    rho = fem.Function(V)
    rho.interpolate(problem.rho)

    n = ufl.FacetNormal(mesh.dolfinx_mesh)
    if mesh.tdim == 2:
        he = ufl.FacetArea(mesh.dolfinx_mesh)
    elif mesh.tdim == 3:
        he = ufl.MaxFacetEdgeLength(mesh.dolfinx_mesh)
    else:
        raise ValueError(f"Unsupported topological dimension: {mesh.tdim}")

    stiffness_form = ufl.dot(rho * ufl.grad(u), ufl.grad(v)) * ufl.dx

    gamma_inside = delta_p2 / he("+") * rho("+") * rho("-") / (rho("+") + rho("-"))
    gamma_boundary = delta_p2 * rho / he
    edge_penalty_form = (
        gamma_inside * ufl.jump(u) * ufl.jump(v) * ufl.dS
        + gamma_boundary * u * v * ufl.ds
    )

    _edge_flux_form = (
        lambda u, v: ufl.dot(
            ufl.jump(u, n),
            rho("+")
            * rho("-")
            / (rho("+") + rho("-"))
            * (ufl.grad(v)("+") + ufl.grad(v)("-")),
        )
        * ufl.dS
        + ufl.dot(u * n, rho * ufl.grad(v)) * ufl.ds
    )
    edge_flux_form = _edge_flux_form(u, v) + _edge_flux_form(v, u)

    exact_form = stiffness_form + edge_penalty_form - edge_flux_form
    exact_form_matrix = fem.assemble_matrix(fem.form(exact_form)).to_scipy()

    if assemble_approximate_form:
        approximate_form = stiffness_form + edge_penalty_form
        approximate_form_matrix = fem.assemble_matrix(
            fem.form(approximate_form)
        ).to_scipy()
    else:
        approximate_form_matrix = None

    f = fem.Function(V)
    f.interpolate(problem.f)
    load_vector = fem.assemble_vector(fem.form(f * v * ufl.dx)).array

    return DiscreteProblem(
        function_space=V,
        exact_form_matrix=exact_form_matrix,
        approximate_form_matrix=approximate_form_matrix,
        load_vector=load_vector,
    )
