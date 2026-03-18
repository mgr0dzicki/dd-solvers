# %%
from meshing import Mesh2D
import matplotlib.pyplot as plt
from problems import Problem, continuous_coefficient_2d, constant_coefficient_2d
from discretization import discretize
import numpy as np
import scipy.sparse as sps
import pandas as pd
import tqdm

# %%
class CG:
    name = "CG"

    def __init__(
        self,
        preconditioner=None,
        rtol: float = 1e-7,
        atol: float = 0,
        maxiter: int | None = 1000,
        estimate_cond: bool = False,
    ):
        self.preconditioner = preconditioner
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.estimate_cond = estimate_cond

    def setup(self, matrix, *args, **kwargs) -> None:
        if self.preconditioner is not None:
            self.preconditioner.setup(matrix, *args, **kwargs)

        self.matrix = matrix

    def solve(self, rhs, x0=None):
        atol = max(self.atol, self.rtol * np.linalg.norm(rhs).item())
        n = len(rhs)
        maxiter = self.maxiter or n * 10
        x = x0 if x0 is not None else np.zeros_like(rhs)
        if self.preconditioner is not None and hasattr(self.preconditioner, "T0"):
            x += self.preconditioner.T0(rhs)
        r = rhs - self.matrix @ x
        rho_prev, p = None, None

        residual_norms = []
        preconditioner_metadata = []
        if self.estimate_cond:
            alphas = []
            betas = []
        for i in range(maxiter):
            res_norm = np.linalg.norm(r).item()
            residual_norms.append(res_norm)
            if res_norm < atol:
                break

            (z, metadata) = (
                self.preconditioner.solve(r)
                if self.preconditioner is not None
                else (r, {})
            )

            preconditioner_metadata.append(metadata)
            rho_cur = np.dot(r, z)
            if i > 0:
                beta = rho_cur / rho_prev
                p *= beta
                p += z
            else:
                p = np.empty_like(r)
                p[:] = z[:]

            q = self.matrix @ p
            alpha = rho_cur / np.dot(p, q)
            x += alpha * p
            r -= alpha * q
            rho_prev = rho_cur

            if self.estimate_cond:
                if i > 0:
                    betas.append(beta)
                alphas.append(alpha)

        metadata = {
            "iterations": len(residual_norms) - 1,
            "residual norms": residual_norms,
            "preconditioner metadata": preconditioner_metadata,
        }

        if self.estimate_cond:
            lmat = np.zeros(
                (len(alphas), len(alphas)),
                dtype=rhs.dtype,
            )
            for i in range(len(alphas)):
                lmat[i, i] = 1 / alphas[i]
                if i > 0:
                    lmat[i, i] += betas[i - 1] / alphas[i - 1]
                    lmat[i - 1, i] = lmat[i, i - 1] = (
                        betas[i - 1] ** (1 / 2) / alphas[i - 1]
                    )
            metadata["condition number estimate"] = np.linalg.cond(lmat, p=2).item()

        if i + 1 == maxiter:
            raise Exception(metadata)

        return x, metadata

    def destroy(self) -> None:
        if self.preconditioner is not None:
            self.preconditioner.destroy()
            self.preconditioner = None
        self.matrix = None

# %%
class GMRES:
    name = "GMRES"

    def __init__(
        self,
        preconditioner=None,
        rtol: float = 1e-7,
        atol: float = 0,
        maxiter: int | None = 1000,
        estimate_cond: bool = False,  # ignored
    ):
        self.preconditioner = preconditioner
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter

    def setup(self, matrix, *args, **kwargs) -> None:
        if self.preconditioner is not None:
            self.preconditioner.setup(matrix, *args, **kwargs)

        self.matrix = matrix

    def solve(self, rhs, x0=None):
        M_x = lambda x: self.preconditioner.solve(x)[0] if self.preconditioner else x
        M = sps.linalg.LinearOperator(self.matrix.shape, matvec=M_x, dtype=rhs.dtype)

        residual_norms = []

        def callback(residual):
            residual_norms.append(np.linalg.norm(residual).item())

        x, info = sps.linalg.gmres(
            self.matrix,
            rhs,
            x0=x0,
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maxiter,
            M=M,
            callback=callback,
            callback_type="legacy",
        )

        metadata = {
            "iterations": len(residual_norms),
            "residual norms": residual_norms,
        }

        if info > 0:
            raise Exception(metadata)

        return x, metadata

    def destroy(self) -> None:
        if self.preconditioner is not None:
            self.preconditioner.destroy()
            self.preconditioner = None
        self.matrix = None

# %%
class OverlappingSchwarz:
    def setup(
        self,
        matrix,
        fine_to_solvers: list[list[int]],
        fine_to_coarse: np.ndarray,
        solvers_to_fine: list[list[int]],
        coarse_to_fine: np.ndarray,
        dofs_per_element: int = 3,
    ):
        self.fine_to_solvers = fine_to_solvers
        self.solvers_to_fine = solvers_to_fine
        self.coarse_to_fine = coarse_to_fine
        self.dofs_per_element = dofs_per_element

        self.n_solvers = len(solvers_to_fine)

        self.solver_dofs = [[] for _ in range(self.n_solvers)]
        self.local_solvers = [None] * self.n_solvers
        for i in range(self.n_solvers):
            for el in solvers_to_fine[i]:
                for d in range(dofs_per_element):
                    self.solver_dofs[i].append(el * dofs_per_element + d)
            self.solver_dofs[i] = np.array(self.solver_dofs[i], dtype=np.int32)
            solver_matrix = matrix[self.solver_dofs[i]][:, self.solver_dofs[i]]
            self.local_solvers[i] = sps.linalg.splu(solver_matrix.tocsc())

        self.matrix = matrix

        matrix_coo = matrix.tocoo()
        coarse_row = fine_to_coarse[matrix_coo.row // dofs_per_element]
        coarse_col = fine_to_coarse[matrix_coo.col // dofs_per_element]
        coarse_data = matrix_coo.data
        coarse_matrix = sps.coo_matrix(
            (coarse_data, (coarse_row, coarse_col)),
            shape=(len(coarse_to_fine), len(coarse_to_fine)),
        ).tocsc()
        self.coarse_solver = sps.linalg.splu(coarse_matrix)

    def destroy(self):
        pass

    def R0(self, x: np.ndarray):
        return (
            x.reshape(-1, self.dofs_per_element)
            .sum(axis=1)[self.coarse_to_fine]
            .sum(axis=1)
        )

    def R0T(self, y_coarse: np.ndarray):
        y_hlp = np.zeros(self.matrix.shape[0] // self.dofs_per_element)
        y_hlp[self.coarse_to_fine] = y_coarse[:, None]
        return y_hlp.repeat(self.dofs_per_element, axis=0)

    def P0(self, x: np.ndarray):
        x_coarse = self.R0(x)
        y_coarse = self.coarse_solver.solve(x_coarse)
        return self.R0T(y_coarse)

    def Pi(self, i: int, x: np.ndarray, out_add: np.ndarray):
        local_x = x[self.solver_dofs[i]]
        local_y = self.local_solvers[i].solve(local_x)
        out_add[self.solver_dofs[i]] += local_y

# %%
class AdditiveSchwarz(OverlappingSchwarz):
    symmetric = True
    name = "Additive"

    def solve(self, x: np.ndarray):
        y = self.P0(x)
        for i in range(self.n_solvers):
            self.Pi(i, x, y)
        return y, {}

# %%
class MultiplicativeSymmetricSchwarz(OverlappingSchwarz):
    symmetric = True
    name = "Multiplicative Symmetric"

    def solve(self, x: np.ndarray):
        y = self.P0(x)
        for i in range(self.n_solvers):
            self.Pi(i, x - self.matrix @ y, y)
        for i in reversed(range(self.n_solvers)):
            self.Pi(i, x - self.matrix @ y, y)
        return y, {}

# %%
class MultiplicativeSchwarz(OverlappingSchwarz):
    symmetric = False
    name = "Multiplicative"

    def solve(self, x: np.ndarray):
        y = self.P0(x)
        for i in range(self.n_solvers):
            self.Pi(i, x - self.matrix @ y, y)
        return y, {}

# %%
class HybridSchwarz(OverlappingSchwarz):
    symmetric = True
    name = "Hybrid"

    def solve(self, x: np.ndarray):
        y = self.P0(x)
        z = x - self.matrix @ y
        u = np.zeros_like(x)
        for i in range(self.n_solvers):
            self.Pi(i, z, u)
        w = u - self.P0(self.matrix @ u)
        return y + w, {}

# %%
preconditioners = [
    AdditiveSchwarz,
    MultiplicativeSymmetricSchwarz,
    MultiplicativeSchwarz,
    HybridSchwarz,
]

# %%
def fine_to_mesh(fine_mesh, n: int, N: int, overlap_n: int):
    v = fine_mesh.vertices[fine_mesh.elements]
    coords = (v.mean(axis=1) * n).astype(int)

    def coords_to_mesh(x, y):
        base_solver = (x // (n // N)) * N + (y // (n // N))
        in_solver_coords = (x % (n // N), y % (n // N))
        solvers = [base_solver]
        if in_solver_coords[1] < overlap_n and base_solver % N != 0:
            solvers.append(base_solver - 1)
        if in_solver_coords[1] >= (n // N - overlap_n) and base_solver % N != N - 1:
            solvers.append(base_solver + 1)
        if in_solver_coords[0] < overlap_n and base_solver >= N:
            solvers.append(base_solver - N)
        if in_solver_coords[0] >= (n // N - overlap_n) and base_solver < N * (N - 1):
            solvers.append(base_solver + N)
        # now we have to check the corners
        if (
            in_solver_coords[0] < overlap_n
            and in_solver_coords[1] < overlap_n
            and base_solver >= N
            and base_solver % N != 0
        ):
            solvers.append(base_solver - N - 1)
        if (
            in_solver_coords[0] < overlap_n
            and in_solver_coords[1] >= (n // N - overlap_n)
            and base_solver >= N
            and base_solver % N != N - 1
        ):
            solvers.append(base_solver - N + 1)
        if (
            in_solver_coords[0] >= (n // N - overlap_n)
            and in_solver_coords[1] < overlap_n
            and base_solver < N * (N - 1)
            and base_solver % N != 0
        ):
            solvers.append(base_solver + N - 1)
        if (
            in_solver_coords[0] >= (n // N - overlap_n)
            and in_solver_coords[1] >= (n // N - overlap_n)
            and base_solver < N * (N - 1)
            and base_solver % N != N - 1
        ):
            solvers.append(base_solver + N + 1)
        return solvers

    return [coords_to_mesh(x, y) for x, y in coords]

# %%
def test(n, N, NN, overlap_n, coefficient="continuous", coefficient_param=None):
    assert n % N == 0
    assert N % NN == 0
    assert overlap_n >= 0 and overlap_n <= n // N // 2

    fine_mesh = Mesh2D.unit_square_uniform(n, n)

    fine_to_solvers = fine_to_mesh(fine_mesh, n, N, overlap_n)
    fine_to_coarse = np.array(fine_to_mesh(fine_mesh, n, NN, 0)).flatten()

    solvers_to_fine = [[] for _ in range(N * N)]
    for i, solvers in enumerate(fine_to_solvers):
        for solver in solvers:
            solvers_to_fine[solver].append(i)

    coarse_to_fine = [[] for _ in range(NN * NN)]
    for i, coarse in enumerate(fine_to_coarse):
        coarse_to_fine[coarse].append(i)
    coarse_to_fine = np.array(coarse_to_fine)

    if coefficient == "constant":
        problem = constant_coefficient_2d.problem
    elif coefficient == "continuous":
        problem = continuous_coefficient_2d.problem
    elif coefficient == "coarse checkerboard":
        problem = Problem(
            delta=7,
            rho=(
                lambda x: ((x[0] * NN).astype(int) + (x[1] * NN).astype(int))
                % 2
                * coefficient_param
                + 1
            ),
            f=constant_coefficient_2d.problem.f,
        )
    elif coefficient == "solvers checkerboard":
        problem = Problem(
            delta=7,
            rho=(
                lambda x: ((x[0] * N).astype(int) + (x[1] * N).astype(int))
                % 2
                * coefficient_param
                + 1
            ),
            f=constant_coefficient_2d.problem.f,
        )

    discrete_problem = discretize(
        problem=problem,
        mesh=fine_mesh,
    )

    results = []
    for preconditioner_cls in preconditioners:
        if preconditioner_cls.symmetric:
            solvers = [GMRES, CG]
        else:
            solvers = [GMRES]

        for solver_cls in solvers:
            preconditioner = preconditioner_cls()
            solver = solver_cls(
                preconditioner=preconditioner,
                rtol=1e-8,
                maxiter=2000,
                estimate_cond=True,
            )

            solver.setup(
                matrix=discrete_problem.exact_form_matrix,
                fine_to_solvers=fine_to_solvers,
                fine_to_coarse=fine_to_coarse,
                solvers_to_fine=solvers_to_fine,
                coarse_to_fine=coarse_to_fine,
                dofs_per_element=3,
            )

            test_metadata = {
                "n": n,
                "N": N,
                "NN": NN,
                "overlap_n": overlap_n,
                "preconditioner": preconditioner_cls.name,
                "solver": solver_cls.name,
                "coefficient": coefficient,
                "coefficient_param": coefficient_param,
            }

            try:
                rhs = np.random.rand(discrete_problem.load_vector.shape[0])
                rhs = (rhs - 0.5) * 2

                sol, metadata = solver.solve(rhs)
                residual_norm = np.linalg.norm(
                    discrete_problem.exact_form_matrix @ sol - rhs
                )
                results.append(
                    {
                        **test_metadata,
                        "residual norm": residual_norm,
                        "iterations": metadata["iterations"],
                        "condition number": metadata.get(
                            "condition number estimate", None
                        ),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        **test_metadata,
                        "exception": str(e),
                    }
                )

            solver.destroy()

    return results

# %%
test_cases = []
for m in range(5, 6):
    for M in range(4, m + 1):
        for MM in range(4, M + 1):
            for coefficient, coefficient_param in [
                ("continuous", None),
                ("constant", None),
                ("coarse checkerboard", 1e6),
                ("solvers checkerboard", 1e6),
            ]:
                n = 2**m
                N = 2**M
                NN = 2**MM
                overlaps = [0]
                for overlap_m in range(m - M):
                    overlaps.append(2**overlap_m)
                for overlap in overlaps:
                    test_cases.append(
                        (n, N, NN, overlap, coefficient, coefficient_param)
                    )

len(test_cases)

# %%
results = []
for test_case in tqdm.tqdm(test_cases):
    results += test(*test_case)

# %%
df = pd.DataFrame(results)
df.to_csv("overlapping_schwarz_results.csv", index=False)
