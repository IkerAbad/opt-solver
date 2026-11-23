"""Classic and generated benchmark problems."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .parsers import read_lp, read_mps
from .types import LPProblem, QPProblem

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "classic"


def beale() -> LPProblem:
    """Beale's cycling example, optimal value -0.05."""
    return read_lp(_DATA_DIR / "beale.lp")


def klee_minty(n: int = 3) -> LPProblem:
    """Klee-Minty cube with ``n`` variables."""
    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i == 0:
            A[0, 0] = 1.0
            b[0] = 5.0
        else:
            A[i, i - 1] = 4.0
            A[i, i] = 1.0
            b[i] = 25.0
    c = np.array([2.0 ** (n - 1 - j) for j in range(n)], dtype=np.float64)
    return LPProblem(
        c=c,
        A=A,
        b=b,
        row_sense=["<="] * n,
        lb=np.zeros(n, dtype=np.float64),
        ub=np.full(n, np.inf, dtype=np.float64),
        sense="max",
        name=f"klee_minty_{n}",
    )


def afiro() -> LPProblem:
    """The classic AFIRO linear program in MPS format."""
    return read_mps(_DATA_DIR / "afiro.mps")


def classic_lp_problems() -> list[LPProblem]:
    problems = [beale(), klee_minty(3), klee_minty(4), afiro()]
    return problems


def generate_lp(
    n: int,
    m: int,
    seed: int,
    *,
    standard_form: bool = False,
) -> LPProblem:
    """Generate a feasible, bounded linear program.

    With ``standard_form=True`` the rows are equalities and the dual is made
    feasible by construction.  Otherwise the rows are ``<=`` inequalities and
    every variable receives a finite upper bound.
    """
    rng = np.random.default_rng(seed)

    if standard_form:
        A = rng.normal(size=(m, n))
        x_feasible = rng.uniform(0.5, 1.5, size=n)
        b = A @ x_feasible
        y = rng.normal(size=m)
        s = rng.uniform(0.5, 1.5, size=n)
        c = A.T @ y + s
        return LPProblem(
            c=c,
            A=A,
            b=b,
            row_sense=["="] * m,
            lb=np.zeros(n, dtype=np.float64),
            ub=np.full(n, np.inf, dtype=np.float64),
            sense="min",
            name=f"gen_eq_n{n}_m{m}_s{seed}",
        )

    A = rng.normal(size=(m, n))
    x_feasible = rng.uniform(0.2, 0.8, size=n)
    slack = rng.uniform(0.2, 1.0, size=m)
    b = A @ x_feasible + slack
    c = rng.normal(size=n)
    return LPProblem(
        c=c,
        A=A,
        b=b,
        row_sense=["<="] * m,
        lb=np.zeros(n, dtype=np.float64),
        ub=np.full(n, 2.0, dtype=np.float64),
        sense="min",
        name=f"gen_box_n{n}_m{m}_s{seed}",
    )


def generate_qp(
    n: int,
    m: int,
    seed: int,
    *,
    box_only: bool = False,
) -> QPProblem:
    """Generate a convex QP, either box-constrained or with general rows."""
    rng = np.random.default_rng(seed)
    factory = rng.normal(size=(n, n))
    Q = factory.T @ factory + 0.1 * np.eye(n)
    target = rng.uniform(-3.0, 3.0, size=n)
    c = -Q @ target
    lb = np.full(n, -1.0, dtype=np.float64)
    ub = np.full(n, 1.0, dtype=np.float64)

    if box_only or m == 0:
        return QPProblem(
            Q=Q,
            c=c,
            A=np.zeros((0, n), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            lb=lb,
            ub=ub,
            name=f"qp_box_n{n}_s{seed}",
        )

    A = rng.normal(size=(m, n))
    x_feasible = rng.uniform(-0.5, 0.5, size=n)
    slack = rng.uniform(0.2, 1.0, size=m)
    b = A @ x_feasible + slack
    return QPProblem(
        Q=Q,
        c=c,
        A=A,
        b=b,
        lb=lb,
        ub=ub,
        name=f"qp_gen_n{n}_m{m}_s{seed}",
    )
