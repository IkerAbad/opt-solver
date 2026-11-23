"""Mehrotra predictor-corrector interior-point method for linear programming."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .standard_form import to_standard_form
from .types import FloatArray, LPProblem, Solution, Status

ClassificationStatus = Literal["feasible", "infeasible", "unbounded", "numerical_error"]
PhaseOneStatus = Literal["feasible", "infeasible", "numerical_error"]


@dataclass
class IPMResult:
    x: FloatArray
    objective: float
    status: Status
    iterations: int
    message: str = ""
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass
class PhaseOneResult:
    status: PhaseOneStatus
    objective: float
    iterations: int
    message: str


@dataclass
class ClassificationResult:
    status: ClassificationStatus
    iterations: int
    message: str
    primal_objective: float
    dual_objective: float


def solve_lp_ipm(
    problem: LPProblem,
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
    store_history: bool = False,
) -> Solution:
    """Solve a linear program with Mehrotra's predictor-corrector method.

    A primal phase I and a dual phase I are solved before the main method.  The
    primal phase I detects infeasibility.  If the primal is feasible but the
    dual phase I is infeasible, the primal is unbounded.  The main
    predictor-corrector is run only when both phase I problems are feasible.

    ``max_iter`` is the limit for each interior-point call, so the total number
    of iterations can be larger.  ``store_history`` records the main phase only.
    """
    std = to_standard_form(problem)
    classification = _classify_standard_lp(
        std.c,
        std.A,
        std.b,
        tol=tol,
        max_iter=max_iter,
    )

    if classification.status == "infeasible":
        return Solution(
            x=np.zeros(problem.n, dtype=np.float64),
            objective=float("nan"),
            status="infeasible",
            iterations=classification.iterations,
            message=classification.message,
        )
    if classification.status == "unbounded":
        return Solution(
            x=np.zeros(problem.n, dtype=np.float64),
            objective=float("nan"),
            status="unbounded",
            iterations=classification.iterations,
            message=classification.message,
        )
    if classification.status == "numerical_error":
        return Solution(
            x=np.zeros(problem.n, dtype=np.float64),
            objective=float("nan"),
            status="numerical_error",
            iterations=classification.iterations,
            message=classification.message,
        )

    result = mehrotra(
        std.c,
        std.A,
        std.b,
        tol=tol,
        max_iter=max_iter,
        store_history=store_history,
    )
    total_iterations = classification.iterations + result.iterations
    message = f"{classification.message} {result.message}"

    if result.status != "optimal":
        return Solution(
            x=np.zeros(problem.n, dtype=np.float64),
            objective=float("nan"),
            status=result.status,
            iterations=total_iterations,
            message=message,
            history=result.history,
        )

    return Solution(
        x=std.recover(result.x),
        objective=std.objective(result.x),
        status="optimal",
        iterations=total_iterations,
        message=message,
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

    if n == 0:
        if np.all(np.abs(b) <= tol):
            return IPMResult(
                x=np.zeros(0, dtype=np.float64),
                objective=0.0,
                status="optimal",
                iterations=0,
                message="No variables and consistent rows.",
            )
        return IPMResult(
            x=np.zeros(0, dtype=np.float64),
            objective=float("nan"),
            status="infeasible",
            iterations=0,
            message="No variables and inconsistent rows.",
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


def _classify_standard_lp(
    c: FloatArray,
    A: FloatArray,
    b: FloatArray,
    *,
    tol: float,
    max_iter: int,
) -> ClassificationResult:
    primal = _primal_phase_one(A, b, tol=tol, max_iter=max_iter)
    if primal.status == "infeasible":
        return ClassificationResult(
            status="infeasible",
            iterations=primal.iterations,
            message=primal.message,
            primal_objective=primal.objective,
            dual_objective=float("nan"),
        )
    if primal.status == "numerical_error":
        return ClassificationResult(
            status="numerical_error",
            iterations=primal.iterations,
            message=primal.message,
            primal_objective=primal.objective,
            dual_objective=float("nan"),
        )

    dual = _dual_phase_one(A, c, tol=tol, max_iter=max_iter)
    if dual.status == "infeasible":
        message = (
            f"Primal phase I is feasible (artificial sum {primal.objective:.2e}). "
            f"Dual phase I is positive (artificial sum {dual.objective:.2e}), "
            "so the primal is unbounded."
        )
        return ClassificationResult(
            status="unbounded",
            iterations=primal.iterations + dual.iterations,
            message=message,
            primal_objective=primal.objective,
            dual_objective=dual.objective,
        )
    if dual.status == "numerical_error":
        message = (
            f"Primal phase I is feasible (artificial sum {primal.objective:.2e}). "
            f"Dual phase I did not converge (artificial sum {dual.objective:.2e})."
        )
        return ClassificationResult(
            status="numerical_error",
            iterations=primal.iterations + dual.iterations,
            message=message,
            primal_objective=primal.objective,
            dual_objective=dual.objective,
        )

    message = (
        "Both phase I problems are feasible "
        f"(primal artificial sum {primal.objective:.2e}, "
        f"dual artificial sum {dual.objective:.2e})."
    )
    return ClassificationResult(
        status="feasible",
        iterations=primal.iterations + dual.iterations,
        message=message,
        primal_objective=primal.objective,
        dual_objective=dual.objective,
    )


def _primal_phase_one(
    A: FloatArray,
    b: FloatArray,
    *,
    tol: float,
    max_iter: int,
) -> PhaseOneResult:
    """Minimize the sum of artificial variables in ``A x + a = b``."""
    A = np.asarray(A, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    m, n = A.shape
    if b.size != m:
        raise ValueError("b must have one entry per row of A")

    phase_tol = _phase_tolerance(tol)
    if m == 0:
        return PhaseOneResult(
            status="feasible",
            objective=0.0,
            iterations=0,
            message="No equality rows, so the primal is feasible.",
        )

    # Equality rows may be multiplied by -1 without changing the feasible set.
    for i in range(m):
        if b[i] < 0.0:
            A[i, :] = -A[i, :]
            b[i] = -b[i]

    A_aug = np.hstack([A, np.eye(m, dtype=np.float64)])
    c_aug = np.concatenate([np.zeros(n, dtype=np.float64), np.ones(m, dtype=np.float64)])
    result = mehrotra(c_aug, A_aug, b, tol=tol, max_iter=max_iter)
    objective = max(0.0, float(result.objective))

    if objective <= phase_tol:
        return PhaseOneResult(
            status="feasible",
            objective=objective,
            iterations=result.iterations,
            message=(
                f"Primal phase I is feasible (artificial sum {objective:.2e}, "
                f"{result.status})."
            ),
        )
    if result.status == "optimal":
        return PhaseOneResult(
            status="infeasible",
            objective=objective,
            iterations=result.iterations,
            message=(
                f"Primal phase I is positive (artificial sum {objective:.2e} "
                f"> {phase_tol:.1e})."
            ),
        )
    return PhaseOneResult(
        status="numerical_error",
        objective=objective,
        iterations=result.iterations,
        message=(
            f"Primal phase I did not converge and its artificial sum is "
            f"{objective:.2e} ({result.status})."
        ),
    )


def _dual_phase_one(
    A: FloatArray,
    c: FloatArray,
    *,
    tol: float,
    max_iter: int,
) -> PhaseOneResult:
    """Minimize the sum of artificial variables in ``A^T y + s - u = c``."""
    A = np.asarray(A, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    m, n = A.shape
    if c.size != n:
        raise ValueError("c must have one entry per column of A")

    phase_tol = _phase_tolerance(tol)
    if n == 0:
        return PhaseOneResult(
            status="feasible",
            objective=0.0,
            iterations=0,
            message="No dual constraints, so the dual is feasible.",
        )

    A_transpose = A.T
    identity = np.eye(n, dtype=np.float64)
    A_dual = np.hstack([A_transpose, -A_transpose, identity, -identity])
    c_dual = np.concatenate(
        [
            np.zeros(2 * m, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
            np.ones(n, dtype=np.float64),
        ]
    )
    result = mehrotra(c_dual, A_dual, c, tol=tol, max_iter=max_iter)
    objective = max(0.0, float(result.objective))

    if objective <= phase_tol:
        return PhaseOneResult(
            status="feasible",
            objective=objective,
            iterations=result.iterations,
            message=(
                f"Dual phase I is feasible (artificial sum {objective:.2e}, "
                f"{result.status})."
            ),
        )
    if result.status == "optimal":
        return PhaseOneResult(
            status="infeasible",
            objective=objective,
            iterations=result.iterations,
            message=(
                f"Dual phase I is positive (artificial sum {objective:.2e} "
                f"> {phase_tol:.1e})."
            ),
        )
    return PhaseOneResult(
        status="numerical_error",
        objective=objective,
        iterations=result.iterations,
        message=(
            f"Dual phase I did not converge and its artificial sum is "
            f"{objective:.2e} ({result.status})."
        ),
    )


def _phase_tolerance(tol: float) -> float:
    return max(1e-8, 10.0 * tol)


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
