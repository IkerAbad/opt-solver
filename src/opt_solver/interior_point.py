"""Mehrotra predictor-corrector interior-point method for linear programming."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .standard_form import to_standard_form
from .types import FloatArray, LPProblem, Solution, Status


@dataclass
class IPMResult:
    x: FloatArray
    objective: float
    status: Status
    iterations: int
    message: str = ""
    history: list[dict[str, float]] = field(default_factory=list)


def solve_lp_ipm(
    problem: LPProblem,
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
    store_history: bool = False,
) -> Solution:
    """Solve a linear program with Mehrotra's predictor-corrector method.

    The method assumes a feasible, bounded standard-form problem.  Infeasible
    and unbounded instances should be classified with :func:`solve_lp_simplex`.
    """
    std = to_standard_form(problem)
    result = mehrotra(
        std.c,
        std.A,
        std.b,
        tol=tol,
        max_iter=max_iter,
        store_history=store_history,
    )

    if result.status != "optimal":
        return Solution(
            x=np.zeros(problem.n, dtype=np.float64),
            objective=float("nan"),
            status=result.status,
            iterations=result.iterations,
            message=result.message,
            history=result.history,
        )

    return Solution(
        x=std.recover(result.x),
        objective=std.objective(result.x),
        status="optimal",
        iterations=result.iterations,
        message=result.message,
        history=result.history,
    )


def mehrotra(
    c: FloatArray,
    A: FloatArray,
    b: FloatArray,
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
    store_history: bool = False,
) -> IPMResult:
    """Run the predictor-corrector method on ``min c @ x``, ``A @ x = b``, ``x >= 0``."""
    c = np.asarray(c, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    m, n = A.shape

    if b.size != m:
        raise ValueError("b must have one entry per row of A")
    if c.size != n:
        raise ValueError("c must have one entry per column of A")

    if m == 0:
        if np.any(c < -tol):
            return IPMResult(
                x=np.zeros(n, dtype=np.float64),
                objective=float("-inf"),
                status="unbounded",
                iterations=0,
                message="Unconstrained objective is unbounded below.",
            )
        return IPMResult(
            x=np.zeros(n, dtype=np.float64),
            objective=0.0,
            status="optimal",
            iterations=0,
            message="No constraints.",
        )

    # Row scaling improves the conditioning of A D A^T without changing x.
    row_norms = np.linalg.norm(A, axis=1)
    scale = np.where(row_norms > 0.0, row_norms, 1.0)
    A = A / scale[:, None]
    b = b / scale

    # Initialization heuristic: reduce the primal and dual residuals with
    # least-squares fits, then shift the iterates into the positive orthant.
    # A plain x = s = 1, y = 0 start can produce enormous intermediate values
    # on problems with large costs, even when the method eventually converges.
    x = np.ones(n, dtype=np.float64)
    s = np.ones(n, dtype=np.float64)
    y = np.zeros(m, dtype=np.float64)
    try:
        y = np.linalg.lstsq(A.T, c, rcond=None)[0]
    except np.linalg.LinAlgError:
        y = np.zeros(m, dtype=np.float64)
    s = c - A.T @ y
    s = s + max(0.0, -float(np.min(s))) + 1.0
    try:
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        x = np.ones(n, dtype=np.float64)
    x = x + max(0.0, -float(np.min(x))) + 1.0
    history: list[dict[str, float]] = []

    for iteration in range(max_iter):
        r_b = A @ x - b
        r_c = A.T @ y + s - c
        mu = float(x @ s / n)
        primal_residual = float(np.linalg.norm(r_b, ord=np.inf))
        dual_residual = float(np.linalg.norm(r_c, ord=np.inf))

        if store_history:
            history.append(
                {
                    "iteration": float(iteration),
                    "mu": mu,
                    "primal_residual": primal_residual,
                    "dual_residual": dual_residual,
                    "objective": float(c @ x),
                }
            )

        if primal_residual <= tol and dual_residual <= tol and mu <= tol:
            return IPMResult(
                x=x,
                objective=float(c @ x),
                status="optimal",
                iterations=iteration,
                message="KKT residuals and complementarity are below tolerance.",
                history=history,
            )

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(s)):
            return IPMResult(
                x=x,
                objective=float("nan"),
                status="numerical_error",
                iterations=iteration,
                message="Iterates became non-finite.",
                history=history,
            )

        D = x / s
        M = A @ (D[:, None] * A.T)

        # Predictor: affine-scaling direction with r_cs = x * s.
        r_cs = x * s
        rhs = -r_b + A @ (r_cs / s) - A @ (D * r_c)
        dy = _solve_spd(M, rhs)
        dx = D * (A.T @ dy + r_c) - r_cs / s
        ds = (-r_cs - s * dx) / x

        alpha_p_aff = _step_length(x, dx)
        alpha_d_aff = _step_length(s, ds)
        mu_aff = float((x + alpha_p_aff * dx) @ (s + alpha_d_aff * ds) / n)
        sigma = (mu_aff / mu) ** 3 if mu > 0.0 else 0.0

        # Corrector: add the second-order complementarity term.
        r_cs = x * s + dx * ds - sigma * mu
        rhs = -r_b + A @ (r_cs / s) - A @ (D * r_c)
        dy = _solve_spd(M, rhs)
        dx = D * (A.T @ dy + r_c) - r_cs / s
        ds = (-r_cs - s * dx) / x

        alpha_p = _step_length(x, dx)
        alpha_d = _step_length(s, ds)

        x = x + alpha_p * dx
        s = s + alpha_d * ds
        y = y + alpha_d * dy

        # Snap tiny negative values to zero.  This is harmless because the
        # step length already keeps the iterates positive up to round-off.
        x = np.maximum(x, 0.0)
        s = np.maximum(s, 0.0)

    return IPMResult(
        x=x,
        objective=float(c @ x),
        status="max_iterations",
        iterations=max_iter,
        message="Reached the iteration limit without meeting the KKT tolerance.",
        history=history,
    )


def _solve_spd(matrix: FloatArray, rhs: FloatArray) -> FloatArray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _step_length(values: FloatArray, direction: FloatArray) -> float:
    negative = direction < 0.0
    if not np.any(negative):
        return 1.0
    ratios = -values[negative] / direction[negative]
    return min(1.0, 0.99 * float(np.min(ratios)))
