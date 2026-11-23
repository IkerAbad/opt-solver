import numpy as np
import pytest

from opt_solver import LPProblem, solve_lp_simplex
from opt_solver.problems import afiro, beale, klee_minty


def test_simple_lp() -> None:
    problem = LPProblem(
        c=np.array([1.0, 1.0]),
        A=np.array([[1.0, 1.0]]),
        b=np.array([1.0]),
        row_sense=[">="],
        lb=np.zeros(2),
        ub=np.full(2, np.inf),
    )
    solution = solve_lp_simplex(problem)

    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(1.0)
    assert np.all(solution.x >= -1e-10)
    assert float(np.sum(solution.x)) == pytest.approx(1.0)


def test_maximization_with_box_bounds() -> None:
    problem = LPProblem(
        c=np.array([3.0, 2.0]),
        A=np.array([[1.0, 1.0]]),
        b=np.array([4.0]),
        row_sense=["<="],
        lb=np.zeros(2),
        ub=np.array([2.0, 3.0]),
        sense="max",
    )
    solution = solve_lp_simplex(problem)

    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(10.0)
    assert solution.x == pytest.approx(np.array([2.0, 2.0]))


def test_free_variable() -> None:
    problem = LPProblem(
        c=np.array([1.0]),
        A=np.array([[1.0]]),
        b=np.array([-1.0]),
        row_sense=[">="],
        lb=np.array([-np.inf]),
        ub=np.array([np.inf]),
    )
    solution = solve_lp_simplex(problem)

    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(-1.0)
    assert solution.x[0] == pytest.approx(-1.0)


def test_equality_with_redundant_row() -> None:
    problem = LPProblem(
        c=np.array([1.0, 1.0]),
        A=np.array([[1.0, 1.0], [2.0, 2.0]]),
        b=np.array([1.0, 2.0]),
        row_sense=["=", "="],
        lb=np.zeros(2),
        ub=np.full(2, np.inf),
    )
    solution = solve_lp_simplex(problem)

    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(1.0)
    assert float(np.sum(solution.x)) == pytest.approx(1.0)


def test_beale_terminates_with_bland() -> None:
    solution = solve_lp_simplex(beale(), tol=1e-10)

    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(-0.05, abs=1e-10)
    assert solution.iterations < 1000


def test_klee_minty_three_variables() -> None:
    solution = solve_lp_simplex(klee_minty(3), tol=1e-10)

    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(45.0, abs=1e-9)


def test_afiro() -> None:
    solution = solve_lp_simplex(afiro(), tol=1e-10)

    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(-464.753142857143, abs=1e-6)
