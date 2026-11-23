"""Convex quadratic programming: active set and projected gradient."""

from __future__ import annotations

import numpy as np

from .simplex import solve_lp_simplex
from .types import FloatArray, LPProblem, QPProblem, Solution


def solve_qp_active_set(
    problem: QPProblem,
    *,
    x0: FloatArray | None = None,
    tol: float = 1e-9,
    max_iter: int = 1000,
) -> Solution:
    """Primal active-set method for ``min 0.5 x'Qx + c'x`` s.t. ``Ax <= b``.

    Variable bounds are converted to inequality rows.  A feasible starting
    point is obtained by solving a phase-I linear program with the simplex
    method.  At each iteration the method solves an equality-constrained QP on
    the current working set, then either releases a constraint with a negative
    multiplier or adds the first blocking constraint.
    """
    n = problem.n
    A_all, b_all = _inequality_system(problem)
    m = A_all.shape[0]

    if x0 is None:
        x = _feasible_point(A_all, b_all, n, tol)
        if x is None:
            return Solution(
                x=np.zeros(n, dtype=np.float64),
                objective=float("nan"),
                status="infeasible",
                iterations=0,
                message="The quadratic program has no feasible point.",
            )
    else:
        x = np.asarray(x0, dtype=np.float64).copy()
        if not _is_feasible(A_all, b_all, x, tol):
            raise ValueError("x0 is not feasible for the quadratic program")

    working_set = [
        i for i in range(m) if abs(float(A_all[i] @ x) - float(b_all[i])) <= 1e-7
    ]

    for iteration in range(max_iter):
        gradient = problem.Q @ x + problem.c

        if working_set:
            A_work = A_all[working_set]
            step, multipliers = _solve_kkt(problem.Q, A_work, gradient)
        else:
            step = _solve_linear(problem.Q, -gradient)
            multipliers = np.zeros(0, dtype=np.float64)

        if np.linalg.norm(step, ord=np.inf) <= tol:
            if not working_set or np.all(multipliers >= -tol):
                return Solution(
                    x=x,
                    objective=_qp_objective(problem.Q, problem.c, x),
                    status="optimal",
                    iterations=iteration,
                    message="All optimality conditions are satisfied.",
                )
            # Release the constraint with the most negative multiplier.
            release = int(np.argmin(multipliers))
            working_set.pop(release)
            continue

        alpha = 1.0
        blocking = -1
        for i in range(m):
            if i in working_set:
                continue
            denominator = float(A_all[i] @ step)
            if denominator > tol:
                ratio = float((b_all[i] - A_all[i] @ x) / denominator)
                if ratio < alpha:
                    alpha = ratio
                    blocking = i

        if alpha < 0.0:
            alpha = 0.0

        x = x + alpha * step
        if blocking >= 0:
            working_set.append(blocking)

    return Solution(
        x=x,
        objective=_qp_objective(problem.Q, problem.c, x),
        status="max_iterations",
        iterations=max_iter,
        message="The active-set method reached the iteration limit.",
    )


def solve_qp_projected_gradient(
    problem: QPProblem,
    *,
    x0: FloatArray | None = None,
    tol: float = 1e-9,
    max_iter: int = 200_000,
) -> Solution:
    """Accelerated projected gradient for box-constrained convex QP.

    The projection onto a box is exact and cheap.  For problems with general
    linear constraints, use :func:`solve_qp_active_set` instead.
    """
    if problem.m > 0:
        raise ValueError("Projected gradient supports box constraints only; use the active-set method for general rows.")

    n = problem.n
    Q = problem.Q
    c = problem.c
    lb = problem.lb
    ub = problem.ub

    lipschitz = float(np.linalg.norm(Q, ord=2))
    if lipschitz <= 0.0:
        lipschitz = 1.0
    step_size = 1.0 / lipschitz

    if x0 is None:
        x = np.clip(np.zeros(n, dtype=np.float64), lb, ub)
    else:
        x = np.clip(np.asarray(x0, dtype=np.float64), lb, ub)

    y = x.copy()
    momentum = 1.0
    f_x = _qp_objective(Q, c, x)

    for iteration in range(max_iter):
        gradient = Q @ y + c
        x_new = np.clip(y - step_size * gradient, lb, ub)
        f_new = _qp_objective(Q, c, x_new)

        # Adaptive restart keeps the accelerated method monotone in practice.
        if f_new > f_x:
            y = x.copy()
            gradient = Q @ y + c
            x_new = np.clip(y - step_size * gradient, lb, ub)
            f_new = _qp_objective(Q, c, x_new)
            momentum = 1.0

        gradient_new = Q @ x_new + c
        projected_gradient = x_new - np.clip(x_new - step_size * gradient_new, lb, ub)
        if (
            np.linalg.norm(projected_gradient, ord=np.inf) <= tol
            and np.linalg.norm(x_new - x, ord=np.inf) <= tol
        ):
            return Solution(
                x=x_new,
                objective=f_new,
                status="optimal",
                iterations=iteration + 1,
                message="The projected gradient is below tolerance.",
            )

        momentum_new = (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        y = x_new + ((momentum - 1.0) / momentum_new) * (x_new - x)
        x = x_new
        f_x = f_new
        momentum = momentum_new

    return Solution(
        x=x,
        objective=_qp_objective(Q, c, x),
        status="max_iterations",
        iterations=max_iter,
        message="The projected-gradient method reached the iteration limit.",
    )


def _inequality_system(problem: QPProblem) -> tuple[FloatArray, FloatArray]:
    n = problem.n
    rows: list[FloatArray] = [problem.A]
    rhs: list[FloatArray] = [problem.b]

    for i in range(n):
        if np.isfinite(problem.ub[i]):
            row = np.zeros((1, n), dtype=np.float64)
            row[0, i] = 1.0
            rows.append(row)
            rhs.append(np.array([problem.ub[i]], dtype=np.float64))
        if np.isfinite(problem.lb[i]):
            row = np.zeros((1, n), dtype=np.float64)
            row[0, i] = -1.0
            rows.append(row)
            rhs.append(np.array([-problem.lb[i]], dtype=np.float64))

    A_all = np.vstack(rows) if rows else np.zeros((0, n), dtype=np.float64)
    b_all = np.concatenate(rhs) if rhs else np.zeros(0, dtype=np.float64)
    return A_all, b_all


def _feasible_point(
    A: FloatArray,
    b: FloatArray,
    n: int,
    tol: float,
) -> FloatArray | None:
    if A.shape[0] == 0:
        return np.zeros(n, dtype=np.float64)

    lp = LPProblem(
        c=np.zeros(n, dtype=np.float64),
        A=A,
        b=b,
        row_sense=["<="] * A.shape[0],
        lb=np.full(n, -np.inf, dtype=np.float64),
        ub=np.full(n, np.inf, dtype=np.float64),
    )
    solution = solve_lp_simplex(lp, tol=tol)
    if solution.status != "optimal":
        return None
    return solution.x


def _is_feasible(A: FloatArray, b: FloatArray, x: FloatArray, tol: float) -> bool:
    if A.shape[0] == 0:
        return True
    return bool(np.all(A @ x <= b + tol))


def _solve_kkt(
    Q: FloatArray,
    A_work: FloatArray,
    gradient: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    n = Q.shape[0]
    p = A_work.shape[0]
    kkt = np.zeros((n + p, n + p), dtype=np.float64)
    kkt[:n, :n] = Q
    kkt[:n, n:] = A_work.T
    kkt[n:, :n] = A_work
    rhs = np.concatenate([-gradient, np.zeros(p, dtype=np.float64)])
    solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    return solution[:n], solution[n:]


def _solve_linear(matrix: FloatArray, rhs: FloatArray) -> FloatArray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _qp_objective(Q: FloatArray, c: FloatArray, x: FloatArray) -> float:
    return float(0.5 * x @ Q @ x + c @ x)
