import copy
import gc
import datetime
from typing import NamedTuple
import numpy as np
import pandas as pd
import tqdm
import torch
import dolfinx
import ufl
from dd_solvers import (
    asm_permutation,
    permute_csr_tensor,
    inverse_permutation,
    csr_to_torch,
    SparseSolver,
)

from .problems import TestCase
from .meshing import MeshFamily
from .discretization import discretize, DiscreteProblem


__all__ = ["Experiment", "ExperimentFactory"]


class Experiment(NamedTuple):
    fine_m: str
    solvers_m: str | None
    coarse_m: str | None
    solver: SparseSolver


class ExperimentFactory:
    def __init__(
        self,
        test_case: TestCase,
        mesh_family: MeshFamily,
        polynomial_degree: int = 1,
        setup_repetitions: int = 3,
        solution_warmup_steps: int = 0,
        solution_measurement_steps: int = 1,
        solution_repetitions: int = 10,
        problem_precision: np.dtype = np.float64,
        error_continuous_norms: tuple[float | int] = (),
        random_rhs: bool = False,
    ):
        self.test_case = test_case
        self.mesh_family = mesh_family
        self.polynomial_degree = polynomial_degree
        self.setup_repetitions = setup_repetitions
        assert self.setup_repetitions >= 1
        self.solution_warmup_steps = solution_warmup_steps
        assert self.solution_warmup_steps >= 0
        self.solution_measurement_steps = solution_measurement_steps
        assert self.solution_measurement_steps >= 1
        self.solution_repetitions = solution_repetitions
        assert self.solution_repetitions >= 1
        self.problem_precision = problem_precision
        self.error_continuous_norms = error_continuous_norms
        self.random_rhs = random_rhs

        self.experiments = []

    def add(self, experiment: Experiment) -> None:
        self.experiments.append(experiment)

    def run(self) -> pd.DataFrame:
        all_fine_ms = set(experiment.fine_m for experiment in self.experiments)
        all_parameters = {}
        for fine_m in tqdm.tqdm(all_fine_ms, desc="Preparing discrete problems"):
            all_parameters[fine_m] = {
                "discrete_problem": self._problem(fine_m),
                "real_solution_fun": self._real_solution_fun(fine_m),
            }

        results = []
        for experiment in tqdm.tqdm(self.experiments, desc="Running experiments"):
            results.append(
                self._repeat_experiment_sandboxed(
                    experiment=experiment,
                    **all_parameters[experiment.fine_m],
                )
            )

        return pd.DataFrame(results)

    def _problem(self, m: str) -> DiscreteProblem:
        discrete_problem = discretize(
            problem=self.test_case.problem,
            mesh=self.mesh_family[m],
            polynomial_degree=self.polynomial_degree,
        )
        return discrete_problem.astype(self.problem_precision)

    def _real_solution_fun(self, m: str) -> dolfinx.fem.Function:
        error_function_space = dolfinx.fem.functionspace(
            self.mesh_family[m].dolfinx_mesh, ("DG", self.polynomial_degree + 2)
        )
        real_solution_fun = dolfinx.fem.Function(error_function_space)
        real_solution_fun.interpolate(self.test_case.solution)
        return real_solution_fun

    def _repeat_experiment_sandboxed(
        self,
        experiment: Experiment,
        discrete_problem: DiscreteProblem,
        real_solution_fun: dolfinx.fem.Function,
    ) -> dict[str, any]:
        results = []
        exception = None
        exception_metadata = None
        for rep in range(self.setup_repetitions):
            try:
                results.append(
                    self._run_experiment(
                        experiment=experiment,
                        discrete_problem=discrete_problem,
                        real_solution_fun=real_solution_fun,
                        measure_solution_time=(rep == self.setup_repetitions - 1),
                    )
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                exception = str(e)
                if hasattr(e, "metadata"):
                    exception_metadata = e.metadata
            finally:
                gc.collect()
                torch.cuda.empty_cache()

            if exception is not None:
                break

        if len(results) == 0:
            combined_result = {}
        else:
            combined_result = {
                k: v for k, v in results[-1].items() if not k.endswith("time")
            }
            time_keys = [key for key in results[-1].keys() if key.endswith("time")]
            times_keys = [key for key in results[-1].keys() if key.endswith("times")]
            for key in time_keys:
                combined_result[f"{key}s"] = [
                    result[key]
                    for result in results
                    if key in result and result[key] is not None
                ]
            for key in times_keys:
                combined_result[key] = sum(
                    [
                        result[key]
                        for result in results
                        if key in result and result[key] is not None
                    ],
                    [],
                )

        return {
            "datetime": datetime.datetime.now(tz=datetime.UTC).isoformat(),
            "test case": self.test_case.name,
            "p": self.polynomial_degree,
            "mesh family": self.mesh_family.name,
            "fine m": experiment.fine_m,
            "solvers m": experiment.solvers_m,
            "coarse m": experiment.coarse_m,
            "solver": str(experiment.solver),
            "random rhs": self.random_rhs,
            "solution warmup steps": self.solution_warmup_steps,
            "solution measurement steps": self.solution_measurement_steps,
            "solution repetitions": self.solution_repetitions,
            "DoFs": discrete_problem.exact_form_matrix.shape[0],
            "matrix nnz": discrete_problem.exact_form_matrix.nnz,
            "metadata": exception_metadata,  # overwritten by `combined_result` if no exception
            **combined_result,
            "exception": exception,
        }

    def _run_experiment(
        self,
        experiment: Experiment,
        discrete_problem: DiscreteProblem,
        real_solution_fun: dolfinx.fem.Function,
        measure_solution_time: bool,
    ) -> dict[str, any]:
        dd_solver = experiment.coarse_m is not None and experiment.solvers_m is not None

        if dd_solver:
            fine_to_solvers_np = self.mesh_family.get_mapping(
                experiment.fine_m, experiment.solvers_m
            )
            solvers_to_coarse_np = self.mesh_family.get_mapping(
                experiment.solvers_m, experiment.coarse_m
            )
        solver = copy.deepcopy(experiment.solver)

        events = [torch.cuda.Event(enable_timing=True) for _ in range(7)]

        events[0].record()
        matrix = csr_to_torch(discrete_problem.exact_form_matrix).cuda()

        if dd_solver:
            fine_to_solvers = torch.as_tensor(fine_to_solvers_np).cuda()
            solvers_to_coarse = torch.as_tensor(solvers_to_coarse_np).cuda()
            fine_to_dofs = torch.as_tensor(
                discrete_problem.function_space.dofmap.list
            ).cuda()

        events[1].record()
        if dd_solver:
            perm = asm_permutation(
                fine_to_solvers=fine_to_solvers,
                solvers_to_coarse=solvers_to_coarse,
                fine_to_dofs=fine_to_dofs,
            )
            inv_perm = inverse_permutation(perm)
            matrix_holder = [matrix]
            del matrix
            matrix = permute_csr_tensor(matrix_holder.pop(), inv_perm)
            gc.collect()

        events[2].record()
        matrix_holder = [matrix]
        del matrix
        if dd_solver:
            n_coarse = solvers_to_coarse.max().item() + 1
            kwargs = dict(
                dofs_per_solver=torch.sum(fine_to_solvers == 0).item()
                * fine_to_dofs.shape[1],
                solvers_per_coarse=solvers_to_coarse.bincount(minlength=n_coarse),
            )
        else:
            kwargs = dict()
        setup_metadata = solver.setup(
            matrix_holder.pop(),
            **kwargs,
            block_size=discrete_problem.function_space.dofmap.list.shape[1],
        )

        events[3].record()
        if self.random_rhs:
            cpu_rhs = np.random.rand(discrete_problem.load_vector.shape[0])
            cpu_rhs = (cpu_rhs - 0.5) * 2
        else:
            cpu_rhs = discrete_problem.load_vector
        rhs = torch.as_tensor(cpu_rhs).cuda()
        if dd_solver:
            rhs = rhs[perm]
        events[4].record()

        if measure_solution_time:
            solve_times = []
            for _ in range(self.solution_repetitions):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                for _ in range(self.solution_warmup_steps):
                    solver.solve(rhs)

                start.record()
                for _ in range(self.solution_measurement_steps):
                    sol, metadata = solver.solve(rhs)
                end.record()

                torch.cuda.synchronize()
                solve_times.append(
                    start.elapsed_time(end) / self.solution_measurement_steps
                )
        else:
            sol, metadata = solver.solve(rhs)
            solve_times = []

        events[5].record()
        if dd_solver:
            sol = sol[inv_perm]
        cpu_sol = sol.cpu().numpy()

        events[6].record()
        torch.cuda.synchronize()

        solver.destroy()

        residual_norm = np.linalg.norm(
            discrete_problem.exact_form_matrix @ cpu_sol - cpu_rhs
        )

        error_norms = {}
        if self.error_continuous_norms:
            la_vec = dolfinx.la.vector(
                discrete_problem.function_space.dofmap.index_map,
                discrete_problem.function_space.dofmap.index_map_bs,
                dtype=cpu_sol.dtype,
            )
            la_vec.array[:] = cpu_sol
            our_solution_fun = dolfinx.fem.Function(
                discrete_problem.function_space, la_vec
            )

            for p in self.error_continuous_norms:
                err_form = dolfinx.fem.form(
                    abs(real_solution_fun - our_solution_fun) ** p * ufl.dx
                )
                local_err = dolfinx.fem.assemble_scalar(err_form)
                error_norms[f"error L{p} norm"] = local_err ** (1 / p)

        return {
            "residual norm": residual_norm,
            **error_norms,
            "metadata": metadata,
            "setup metadata": setup_metadata,
            "matrix and DOF map copy time": events[0].elapsed_time(events[1]),
            "matrix permute time": events[1].elapsed_time(events[2]),
            "solver setup time": events[2].elapsed_time(events[3]),
            "rhs copy and permute time": events[3].elapsed_time(events[4]),
            "solve times": solve_times,
            "solution permute and copy time": events[5].elapsed_time(events[6]),
        }
