"""Benchmark helpers and a thin HiGHS reference wrapper."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from .interior_point import solve_lp_ipm
from .qp import solve_qp_active_set, solve_qp_projected_gradient
from .simplex import solve_lp_simplex
from .types import LPProblem, QPProblem, Solution


class HighsReference:
    """Solve LP and QP models with HiGHS through ``highspy``."""

    def __init__(self) -> None:
        import highspy

        self.highspy = highspy

    def _new_model(self) -> Any:
        highs = self.highspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.setOptionValue("presolve", "off")
        highs.setOptionValue("primal_feasibility_tolerance", 1e-10)
        highs.setOptionValue("dual_feasibility_tolerance", 1e-10)
        highs.setOptionValue("ipm_optimality_tolerance", 1e-10)
        return highs

    def solve_lp(self, problem: LPProblem, *, method: str = "simplex") -> dict[str, Any]:
        highs = self._new_model()
        if method in {"simplex", "ipm"}:
            highs.setOptionValue("solver", method)

        lower = np.where(np.isfinite(problem.lb), problem.lb, -self.highspy.kHighsInf)
        upper = np.where(np.isfinite(problem.ub), problem.ub, self.highspy.kHighsInf)
        highs.addVars(problem.n, lower, upper)
        variables = highs.getVariables()

        if np.any(problem.c != 0.0):
            objective = highs.expr()
            for j, value in enumerate(problem.c):
                if value != 0.0:
                    objective += float(value) * variables[j]
            if problem.sense == "min":
                highs.minimize(objective)
            else:
                highs.maximize(objective)
        elif problem.sense == "max":
            highs.setMaximize()
        else:
            highs.setMinimize()

        if problem.constant != 0.0:
            highs.changeObjectiveOffset(problem.constant)

        infeasible = False
        for i in range(problem.m):
            nonzero = np.flatnonzero(problem.A[i] != 0.0)
            if nonzero.size == 0:
                rhs = float(problem.b[i])
                sense = problem.row_sense[i]
                if (sense == "<=" and rhs < 0.0) or (sense == ">=" and rhs > 0.0) or (
                    sense == "=" and rhs != 0.0
                ):
                    infeasible = True
                    break
                continue
            row_expression = highs.expr()
            for column in nonzero:
                row_expression += float(problem.A[i, column]) * variables[int(column)]
            if problem.row_sense[i] == "<=":
                highs.addConstr(row_expression <= float(problem.b[i]))
            elif problem.row_sense[i] == ">=":
                highs.addConstr(row_expression >= float(problem.b[i]))
            else:
                highs.addConstr(row_expression == float(problem.b[i]))

        if infeasible:
            return _reference_result("infeasible", np.zeros(problem.n), 0, 0.0)

        start = time.perf_counter()
        highs.optimize()
        elapsed = time.perf_counter() - start

        status = _map_status(highs.getModelStatus())
        solution = highs.getSolution()
        x = (
            np.asarray(solution.col_value, dtype=np.float64)
            if status == "optimal"
            else np.zeros(problem.n, dtype=np.float64)
        )
        info = highs.getInfo()
        iterations = int(
            info.simplex_iteration_count
            + info.ipm_iteration_count
            + info.qp_iteration_count
        )
        objective = float(highs.getObjectiveValue()) if status == "optimal" else float("nan")
        return _reference_result(status, x, iterations, elapsed, objective)

    def solve_qp(self, problem: QPProblem, *, method: str = "default") -> dict[str, Any]:
        highs = self._new_model()
        if method == "ipm":
            highs.setOptionValue("solver", "ipm")

        lower = np.where(np.isfinite(problem.lb), problem.lb, -self.highspy.kHighsInf)
        upper = np.where(np.isfinite(problem.ub), problem.ub, self.highspy.kHighsInf)
        highs.addVars(problem.n, lower, upper)
        variables = highs.getVariables()

        if np.any(problem.c != 0.0):
            objective = highs.expr()
            for j, value in enumerate(problem.c):
                if value != 0.0:
                    objective += float(value) * variables[j]
            highs.minimize(objective)
        else:
            highs.setMinimize()

        hessian = self.highspy.HighsHessian()
        hessian.dim_ = problem.n
        hessian.format_ = self.highspy.HessianFormat.kTriangular
        starts = [0]
        indices: list[int] = []
        values: list[float] = []
        for j in range(problem.n):
            for i in range(j, problem.n):
                value = problem.Q[i, j]
                if value != 0.0:
                    indices.append(i)
                    values.append(float(value))
            starts.append(len(indices))
        hessian.start_ = np.asarray(starts, dtype=np.int32)
        hessian.index_ = np.asarray(indices, dtype=np.int32)
        hessian.value_ = np.asarray(values, dtype=np.float64)
        highs.passHessian(hessian)

        for i in range(problem.m):
            nonzero = np.flatnonzero(problem.A[i] != 0.0)
            if nonzero.size == 0:
                if float(problem.b[i]) < 0.0:
                    return _reference_result("infeasible", np.zeros(problem.n), 0, 0.0)
                continue
            row_expression = highs.expr()
            for column in nonzero:
                row_expression += float(problem.A[i, column]) * variables[int(column)]
            highs.addConstr(row_expression <= float(problem.b[i]))

        start = time.perf_counter()
        highs.optimize()
        elapsed = time.perf_counter() - start

        status = _map_status(highs.getModelStatus())
        solution = highs.getSolution()
        x = (
            np.asarray(solution.col_value, dtype=np.float64)
            if status == "optimal"
            else np.zeros(problem.n, dtype=np.float64)
        )
        info = highs.getInfo()
        iterations = int(
            info.qp_iteration_count
            + info.ipm_iteration_count
            + info.simplex_iteration_count
        )
        objective = float(highs.getObjectiveValue()) if status == "optimal" else float("nan")
        return _reference_result(status, x, iterations, elapsed, objective)


def benchmark_lp(
    problems: list[LPProblem],
    *,
    tolerance: float = 1e-8,
    solver_tol: float = 1e-10,
) -> list[dict[str, Any]]:
    reference = HighsReference()
    rows: list[dict[str, Any]] = []

    for problem in problems:
        reference_simplex = reference.solve_lp(problem, method="simplex")
        reference_ipm = reference.solve_lp(problem, method="ipm")

        start = time.perf_counter()
        simplex_solution = solve_lp_simplex(problem, tol=solver_tol)
        simplex_time = time.perf_counter() - start
        rows.append(
            _lp_row(
                problem,
                "simplex",
                simplex_solution,
                reference_simplex,
                simplex_time,
                tolerance,
            )
        )

        start = time.perf_counter()
        ipm_solution = solve_lp_ipm(problem, tol=solver_tol)
        ipm_time = time.perf_counter() - start
        rows.append(
            _lp_row(problem, "ipm", ipm_solution, reference_ipm, ipm_time, tolerance)
        )

    return rows


def benchmark_qp(
    problems: list[QPProblem],
    *,
    tolerance: float = 1e-8,
    solver_tol: float = 1e-9,
    projected_gradient_max_iter: int = 200_000,
) -> list[dict[str, Any]]:
    reference = HighsReference()
    rows: list[dict[str, Any]] = []

    for problem in problems:
        reference_solution = reference.solve_qp(problem)

        start = time.perf_counter()
        active_solution = solve_qp_active_set(problem, tol=solver_tol)
        active_time = time.perf_counter() - start
        rows.append(
            _qp_row(
                problem,
                "active_set",
                active_solution,
                reference_solution,
                active_time,
                tolerance,
            )
        )

        if problem.m == 0:
            start = time.perf_counter()
            projected_solution = solve_qp_projected_gradient(
                problem,
                tol=solver_tol,
                max_iter=projected_gradient_max_iter,
            )
            projected_time = time.perf_counter() - start
            rows.append(
                _qp_row(
                    problem,
                    "projected_gradient",
                    projected_solution,
                    reference_solution,
                    projected_time,
                    tolerance,
                )
            )

    return rows


def format_summary(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'problem':<28} {'kind':<4} {'solver':<19} {'status':<12} "
        f"{'objective':>15} {'iters':>7} {'time':>10} {'gap':>10}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        objective = float(row["objective"])
        gap = float(row["gap"])
        lines.append(
            f"{str(row['problem']):<28} {str(row['kind']):<4} {str(row['solver']):<19} "
            f"{str(row['status']):<12} {objective:>15.8g} {int(row['iterations']):>7d} "
            f"{float(row['time']):>10.6f} {gap:>10.2e}"
        )
    return "\n".join(lines)


def format_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Problem | Kind | Solver | Objective | Iterations | Time (s) | Gap |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['problem']} | {row['kind']} | {row['solver']} | "
            f"{float(row['objective']):.10g} | {int(row['iterations'])} | "
            f"{float(row['time']):.6f} | {float(row['gap']):.2e} |"
        )
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _reference_result(
    status: str,
    x: np.ndarray,
    iterations: int,
    elapsed: float,
    objective: float = float("nan"),
) -> dict[str, Any]:
    return {
        "status": status,
        "objective": objective,
        "iterations": iterations,
        "time": elapsed,
        "x": x,
    }


def _map_status(status: object) -> str:
    name = str(status)
    if name.endswith("kOptimal"):
        return "optimal"
    if name.endswith("kInfeasible"):
        return "infeasible"
    if name.endswith("kUnbounded"):
        return "unbounded"
    if name.endswith("kIterationLimit") or name.endswith("kTimeLimit"):
        return "max_iterations"
    return "numerical_error"


def _lp_row(
    problem: LPProblem,
    solver_name: str,
    solution: Solution,
    reference: dict[str, Any],
    elapsed: float,
    tolerance: float,
) -> dict[str, Any]:
    gap = float("nan")
    if solution.status == "optimal" and reference["status"] == "optimal":
        gap = abs(solution.objective - float(reference["objective"]))
    return {
        "problem": problem.name,
        "kind": "LP",
        "n": problem.n,
        "m": problem.m,
        "solver": solver_name,
        "status": solution.status,
        "objective": solution.objective,
        "iterations": solution.iterations,
        "time": elapsed,
        "reference_objective": reference["objective"],
        "reference_iterations": reference["iterations"],
        "reference_time": reference["time"],
        "gap": gap,
        "within_tolerance": bool(np.isfinite(gap) and gap <= tolerance),
    }


def _qp_row(
    problem: QPProblem,
    solver_name: str,
    solution: Solution,
    reference: dict[str, Any],
    elapsed: float,
    tolerance: float,
) -> dict[str, Any]:
    gap = float("nan")
    if solution.status == "optimal" and reference["status"] == "optimal":
        gap = abs(solution.objective - float(reference["objective"]))
    return {
        "problem": problem.name,
        "kind": "QP",
        "n": problem.n,
        "m": problem.m,
        "solver": solver_name,
        "status": solution.status,
        "objective": solution.objective,
        "iterations": solution.iterations,
        "time": elapsed,
        "reference_objective": reference["objective"],
        "reference_iterations": reference["iterations"],
        "reference_time": reference["time"],
        "gap": gap,
        "within_tolerance": bool(np.isfinite(gap) and gap <= tolerance),
    }
