"""Shared data structures for optimization problems and solutions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Sense = Literal["min", "max"]
RowSense = Literal["<=", ">=", "="]
Status = Literal["optimal", "infeasible", "unbounded", "max_iterations", "numerical_error"]


def _as_float_array(value: object, ndim: int, name: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {arr.shape}")
    return arr


@dataclass
class LPProblem:
    """A linear program in general form.

    Minimize or maximize ``c @ x`` subject to rows with individual senses and
    variable bounds ``lb <= x <= ub``.  Infinite bounds are represented with
    ``np.inf`` and ``-np.inf``.
    """

    c: FloatArray
    A: FloatArray
    b: FloatArray
    row_sense: list[RowSense]
    lb: FloatArray
    ub: FloatArray
    sense: Sense = "min"
    name: str = ""
    var_names: list[str] = field(default_factory=list)
    row_names: list[str] = field(default_factory=list)
    constant: float = 0.0

    def __post_init__(self) -> None:
        self.c = _as_float_array(self.c, 1, "c")
        self.A = _as_float_array(self.A, 2, "A")
        self.b = _as_float_array(self.b, 1, "b")
        self.lb = _as_float_array(self.lb, 1, "lb")
        self.ub = _as_float_array(self.ub, 1, "ub")

        n = self.c.size
        if self.A.shape[1] != n:
            raise ValueError("A must have one column per variable")
        if self.A.shape[0] != self.b.size:
            raise ValueError("A must have one row per entry of b")
        if self.lb.size != n or self.ub.size != n:
            raise ValueError("lb and ub must have one entry per variable")
        if len(self.row_sense) != self.b.size:
            raise ValueError("row_sense must have one entry per row")
        if np.any(self.lb > self.ub):
            raise ValueError("lb must be <= ub")
        if self.sense not in ("min", "max"):
            raise ValueError("sense must be 'min' or 'max'")
        if any(s not in ("<=", ">=", "=") for s in self.row_sense):
            raise ValueError("row_sense entries must be '<=', '>=', or '='")

    @property
    def n(self) -> int:
        return int(self.c.size)

    @property
    def m(self) -> int:
        return int(self.b.size)


@dataclass
class QPProblem:
    """A convex quadratic program.

    Minimize ``0.5 * x @ Q @ x + c @ x`` subject to ``A @ x <= b`` and
    ``lb <= x <= ub``.  ``Q`` must be symmetric positive semidefinite.
    """

    Q: FloatArray
    c: FloatArray
    A: FloatArray
    b: FloatArray
    lb: FloatArray
    ub: FloatArray
    name: str = ""

    def __post_init__(self) -> None:
        self.Q = _as_float_array(self.Q, 2, "Q")
        self.c = _as_float_array(self.c, 1, "c")
        self.A = _as_float_array(self.A, 2, "A")
        self.b = _as_float_array(self.b, 1, "b")
        self.lb = _as_float_array(self.lb, 1, "lb")
        self.ub = _as_float_array(self.ub, 1, "ub")

        n = self.c.size
        if self.Q.shape != (n, n):
            raise ValueError("Q must be square with one row per variable")
        if self.A.shape[1] != n:
            raise ValueError("A must have one column per variable")
        if self.A.shape[0] != self.b.size:
            raise ValueError("A must have one row per entry of b")
        if self.lb.size != n or self.ub.size != n:
            raise ValueError("lb and ub must have one entry per variable")
        if np.any(self.lb > self.ub):
            raise ValueError("lb must be <= ub")
        if not np.allclose(self.Q, self.Q.T, atol=1e-10):
            raise ValueError("Q must be symmetric")
        if np.any(np.linalg.eigvalsh(self.Q) < -1e-10):
            raise ValueError("Q must be positive semidefinite")

    @property
    def n(self) -> int:
        return int(self.c.size)

    @property
    def m(self) -> int:
        return int(self.b.size)


@dataclass
class Solution:
    """Result returned by every solver in the package."""

    x: FloatArray
    objective: float
    status: Status
    iterations: int
    message: str = ""
    dual: FloatArray | None = None
    slack: FloatArray | None = None
    history: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)


@dataclass
class StandardLP:
    """Standard form ``min c @ x`` s.t. ``A @ x = b``, ``x >= 0``.

    ``x_original = T @ x_standard + d``.  The objective reported to the user is
    ``obj_sign * (c @ x_standard + obj_const)``.
    """

    c: FloatArray
    A: FloatArray
    b: FloatArray
    T: FloatArray
    d: FloatArray
    obj_sign: float
    obj_const: float
    n_orig: int
    n_struct: int

    def recover(self, x_std: FloatArray) -> FloatArray:
        return self.T @ x_std + self.d

    def objective(self, x_std: FloatArray) -> float:
        return float(self.obj_sign * (self.c @ x_std + self.obj_const))
