"""Conversion of a general linear program to standard form."""

from __future__ import annotations

import numpy as np

from .types import FloatArray, LPProblem, RowSense, StandardLP


def to_standard_form(problem: LPProblem) -> StandardLP:
    """Convert ``problem`` to ``min c @ x`` s.t. ``A @ x = b``, ``x >= 0``.

    Free variables are split into two non-negative parts, finite lower bounds
    are shifted, finite upper bounds become extra rows, and inequality rows
    receive slack or surplus variables.  Every row is scaled so that ``b >= 0``.
    """
    n = problem.n
    c_eff = -problem.c if problem.sense == "max" else problem.c
    obj_sign = -1.0 if problem.sense == "max" else 1.0

    structural_columns: list[FloatArray] = []
    d = np.zeros(n, dtype=np.float64)
    box_rows: list[tuple[int, float]] = []

    for j in range(n):
        lb = float(problem.lb[j])
        ub = float(problem.ub[j])
        column = np.zeros(n, dtype=np.float64)

        if np.isfinite(lb) and np.isfinite(ub):
            width = ub - lb
            if width <= 1e-12:
                d[j] = lb
                continue
            column[j] = 1.0
            structural_columns.append(column)
            d[j] = lb
            box_rows.append((len(structural_columns) - 1, width))
        elif np.isfinite(lb):
            column[j] = 1.0
            structural_columns.append(column)
            d[j] = lb
        elif np.isfinite(ub):
            column[j] = -1.0
            structural_columns.append(column)
            d[j] = ub
        else:
            plus = np.zeros(n, dtype=np.float64)
            plus[j] = 1.0
            structural_columns.append(plus)
            minus = np.zeros(n, dtype=np.float64)
            minus[j] = -1.0
            structural_columns.append(minus)

    n_struct = len(structural_columns)
    if structural_columns:
        T_struct = np.column_stack(structural_columns)
    else:
        T_struct = np.zeros((n, 0), dtype=np.float64)

    rows: list[FloatArray] = []
    rhs: list[float] = []
    senses: list[RowSense] = []

    for i in range(problem.m):
        rows.append(problem.A[i] @ T_struct)
        rhs.append(float(problem.b[i] - problem.A[i] @ d))
        senses.append(problem.row_sense[i])

    for k, width in box_rows:
        row = np.zeros(n_struct, dtype=np.float64)
        row[k] = 1.0
        rows.append(row)
        rhs.append(width)
        senses.append("<=")

    # Equality rows can be multiplied by -1 freely.  For inequalities the
    # sense must be flipped when the right-hand side changes sign.
    for i in range(len(rows)):
        if rhs[i] < 0.0:
            rows[i] = -rows[i]
            rhs[i] = -rhs[i]
            if senses[i] == "<=":
                senses[i] = ">="
            elif senses[i] == ">=":
                senses[i] = "<="

    m = len(rows)
    n_slack = sum(1 for sense in senses if sense != "=")
    A = np.zeros((m, n_struct + n_slack), dtype=np.float64)
    if m:
        A[:, :n_struct] = np.vstack(rows)

    slack = 0
    for i, sense in enumerate(senses):
        if sense == "<=":
            A[i, n_struct + slack] = 1.0
            slack += 1
        elif sense == ">=":
            A[i, n_struct + slack] = -1.0
            slack += 1

    b = np.asarray(rhs, dtype=np.float64)
    T = np.hstack([T_struct, np.zeros((n, n_slack), dtype=np.float64)])
    c_std = T.T @ c_eff

    return StandardLP(
        c=np.asarray(c_std, dtype=np.float64),
        A=A,
        b=b,
        T=T,
        d=d,
        obj_sign=obj_sign,
        obj_const=float(c_eff @ d + obj_sign * problem.constant),
        n_orig=n,
        n_struct=n_struct,
    )
