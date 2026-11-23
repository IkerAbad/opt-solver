"""Two-phase primal simplex with Bland's anti-cycling rule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .standard_form import to_standard_form
from .types import FloatArray, LPProblem, Solution, Status


@dataclass
class SimplexResult:
    x: FloatArray
    objective: float
    status: Status
    iterations: int
    basis: list[int]
    message: str = ""


def solve_lp_simplex(
    problem: LPProblem,
    *,
    tol: float = 1e-9,
    max_iter: int = 10000,
) -> Solution:
    """Solve a linear program with the two-phase primal simplex method."""
    std = to_standard_form(problem)
    result = two_phase_simplex(std.c, std.A, std.b, tol=tol, max_iter=max_iter)

    if result.status != "optimal":
        return Solution(
            x=np.zeros(problem.n, dtype=np.float64),
            objective=float("nan"),
            status=result.status,
            iterations=result.iterations,
            message=result.message,
        )

    return Solution(
        x=std.recover(result.x),
        objective=std.objective(result.x),
        status="optimal",
        iterations=result.iterations,
        message=result.message,
    )


def two_phase_simplex(
    c: FloatArray,
    A: FloatArray,
    b: FloatArray,
    *,
    tol: float = 1e-9,
    max_iter: int = 10000,
) -> SimplexResult:
    """Solve ``min c @ x`` s.t. ``A @ x = b``, ``x >= 0``.

    The input is copied and normalized so that ``b >= 0``.  Phase I minimizes
    the sum of artificial variables.  If its optimal value is positive the
    original system is infeasible.  Phase II optimizes the real objective from
    the feasible basis produced by phase I.
    """
    c = np.asarray(c, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    m, n = A.shape

    if b.size != m:
        raise ValueError("b must have one entry per row of A")
    if c.size != n:
        raise ValueError("c must have one entry per column of A")

    # In equality form, multiplying a row by -1 leaves the feasible set intact.
    for i in range(m):
        if b[i] < 0.0:
            A[i, :] = -A[i, :]
            b[i] = -b[i]

    if m == 0:
        if np.any(c < -tol):
            return SimplexResult(
                x=np.zeros(n, dtype=np.float64),
                objective=float("-inf"),
                status="unbounded",
                iterations=0,
                basis=[],
                message="Unconstrained objective is unbounded below.",
            )
        return SimplexResult(
            x=np.zeros(n, dtype=np.float64),
            objective=0.0,
            status="optimal",
            iterations=0,
            basis=[],
            message="No constraints.",
        )

    # Phase I.
    A1 = np.hstack([A, np.eye(m, dtype=np.float64)])
    c1 = np.concatenate([np.zeros(n, dtype=np.float64), np.ones(m, dtype=np.float64)])
    basis1 = list(range(n, n + m))
    tab, basis, status1, iterations1 = _simplex_tableau(A1, b, c1, basis1, tol, max_iter)

    if status1 == "unbounded":
        return SimplexResult(
            x=np.zeros(n, dtype=np.float64),
            objective=float("nan"),
            status="numerical_error",
            iterations=iterations1,
            basis=basis,
            message="Phase I reported an unbounded auxiliary problem.",
        )
    if status1 == "max_iterations":
        return SimplexResult(
            x=np.zeros(n, dtype=np.float64),
            objective=float("nan"),
            status="max_iterations",
            iterations=iterations1,
            basis=basis,
            message="Phase I reached the iteration limit.",
        )

    phase1_value = -float(tab[m, n + m])
    infeasibility_tol = max(1e-8, 10.0 * tol)
    if phase1_value > infeasibility_tol:
        return SimplexResult(
            x=np.zeros(n, dtype=np.float64),
            objective=float("nan"),
            status="infeasible",
            iterations=iterations1,
            basis=basis,
            message=f"Phase I optimum is {phase1_value:.3e}, above tolerance.",
        )

    # Try to pivot every artificial variable out of the basis.  A row whose
    # artificial variable cannot be removed is redundant and is dropped.
    rows_to_keep: list[int] = []
    initial_basis: list[int] = []
    for i in range(m):
        if basis[i] >= n:
            for j in range(n):
                if j not in basis and abs(tab[i, j]) > 1e-10:
                    _pivot(tab, i, j)
                    basis[i] = j
                    break

        if basis[i] < n:
            rows_to_keep.append(i)
            initial_basis.append(basis[i])
        elif abs(tab[i, n + m]) > infeasibility_tol:
            return SimplexResult(
                x=np.zeros(n, dtype=np.float64),
                objective=float("nan"),
                status="infeasible",
                iterations=iterations1,
                basis=basis,
                message="A redundant row has a non-zero right-hand side.",
            )

    if not rows_to_keep:
        if np.any(c < -tol):
            return SimplexResult(
                x=np.zeros(n, dtype=np.float64),
                objective=float("-inf"),
                status="unbounded",
                iterations=iterations1,
                basis=[],
                message="All constraints are redundant and the objective is unbounded.",
            )
        return SimplexResult(
            x=np.zeros(n, dtype=np.float64),
            objective=0.0,
            status="optimal",
            iterations=iterations1,
            basis=[],
            message="All constraints are redundant.",
        )

    A2 = A[rows_to_keep, :]
    b2 = b[rows_to_keep]
    remaining_iter = max(1, max_iter - iterations1)
    tab2, basis2, status2, iterations2 = _simplex_tableau(
        A2, b2, c, initial_basis, tol, remaining_iter
    )

    x = np.zeros(n, dtype=np.float64)
    if basis2:
        x[basis2] = tab2[: len(basis2), n]
    objective = float(c @ x)

    if status2 != "optimal":
        return SimplexResult(
            x=x,
            objective=float("nan"),
            status=status2,
            iterations=iterations1 + iterations2,
            basis=basis2,
            message="Phase II did not reach optimality.",
        )

    return SimplexResult(
        x=x,
        objective=objective,
        status="optimal",
        iterations=iterations1 + iterations2,
        basis=basis2,
        message="Optimal solution found.",
    )


def _build_tableau(
    A: FloatArray,
    b: FloatArray,
    c: FloatArray,
    basis: list[int],
) -> FloatArray:
    """Build the canonical simplex tableau for the given basis."""
    m, n = A.shape
    tab = np.zeros((m + 1, n + 1), dtype=np.float64)
    if m == 0:
        tab[0, :n] = c
        return tab

    B = A[:, basis]
    try:
        BinvA = np.linalg.solve(B, A)
        Binvb = np.linalg.solve(B, b)
    except np.linalg.LinAlgError:
        BinvA = np.linalg.lstsq(B, A, rcond=None)[0]
        Binvb = np.linalg.lstsq(B, b, rcond=None)[0]

    tab[:m, :n] = BinvA
    tab[:m, n] = Binvb
    cB = c[basis]
    tab[m, :n] = c - cB @ BinvA
    tab[m, n] = -float(cB @ Binvb)
    return tab


def _pivot(tab: FloatArray, row: int, column: int) -> None:
    pivot = tab[row, column]
    if abs(pivot) < 1e-15:
        raise ValueError("numerical pivot is too small")
    tab[row, :] /= pivot
    for i in range(tab.shape[0]):
        if i == row:
            continue
        factor = tab[i, column]
        if factor != 0.0:
            tab[i, :] -= factor * tab[row, :]


def _simplex_tableau(
    A: FloatArray,
    b: FloatArray,
    c: FloatArray,
    basis: list[int],
    tol: float,
    max_iter: int,
) -> tuple[FloatArray, list[int], Status, int]:
    """Run the primal simplex method on an equality-form problem."""
    m, n = A.shape
    if len(basis) != m:
        raise ValueError("basis must have one entry per row of A")

    if m == 0:
        tab = np.zeros((1, n + 1), dtype=np.float64)
        tab[0, :n] = c
        if np.any(c < -tol):
            return tab, basis, "unbounded", 0
        return tab, basis, "optimal", 0

    tab = _build_tableau(A, b, c, basis)
    for iteration in range(max_iter):
        nonbasic = np.ones(n, dtype=bool)
        nonbasic[basis] = False
        candidates = np.flatnonzero((tab[m, :n] < -tol) & nonbasic)
        if candidates.size == 0:
            return tab, basis, "optimal", iteration

        # Bland's rule: smallest index with a negative reduced cost.
        entering = int(candidates[0])
        direction = tab[:m, entering]
        positive = direction > tol
        if not np.any(positive):
            return tab, basis, "unbounded", iteration

        ratios = np.full(m, np.inf, dtype=np.float64)
        ratios[positive] = tab[:m, n][positive] / direction[positive]
        min_ratio = float(np.min(ratios[positive]))
        tie_tol = 1e-12 * max(1.0, abs(min_ratio))
        tied = np.flatnonzero(positive & (np.abs(ratios - min_ratio) <= tie_tol))

        # Bland's rule: smallest basic-variable index among tied rows.
        leaving = int(tied[np.argmin([basis[i] for i in tied])])
        _pivot(tab, leaving, entering)
        basis[leaving] = entering

        # Refactor periodically to stop numerical drift in the dense tableau.
        if (iteration + 1) % 50 == 0:
            tab = _build_tableau(A, b, c, basis)

    return tab, basis, "max_iterations", max_iter
