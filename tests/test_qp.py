import numpy as np
import pytest

from opt_solver import QPProblem, solve_qp_active_set, solve_qp_projected_gradient


def test_box_qp_known_solution() -> None:
    problem = QPProblem(
        Q=np.eye(2),
        c=np.array([-2.0, 1.0]),
        A=np.zeros((0, 2)),
        b=np.zeros(0),
        lb=np.zeros(2),
        ub=np.ones(2),
    )

    active = solve_qp_active_set(problem)
    projected = solve_qp_projected_gradient(problem)

    assert active.status == "optimal"
    assert projected.status == "optimal"
    assert active.objective == pytest.approx(-1.5, abs=1e-9)
    assert projected.objective == pytest.approx(-1.5, abs=1e-9)
    assert active.x == pytest.approx(np.array([1.0, 0.0]), abs=1e-7)


def test_general_qp_known_solution() -> None:
    problem = QPProblem(
        Q=np.eye(2),
        c=np.zeros(2),
        A=np.array([[-1.0, -1.0]]),
        b=np.array([-1.0]),
        lb=np.full(2, -np.inf),
        ub=np.full(2, np.inf),
    )
    solution = solve_qp_active_set(problem)

    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(0.25, abs=1e-10)
    assert solution.x == pytest.approx(np.array([0.5, 0.5]), abs=1e-7)


def test_equality_as_two_inequalities() -> None:
    problem = QPProblem(
        Q=np.eye(2),
        c=np.zeros(2),
        A=np.array([[1.0, 1.0], [-1.0, -1.0]]),
        b=np.array([1.0, -1.0]),
        lb=np.full(2, -np.inf),
        ub=np.full(2, np.inf),
    )
    solution = solve_qp_active_set(problem)

    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(0.25, abs=1e-10)
    assert solution.x == pytest.approx(np.array([0.5, 0.5]), abs=1e-7)


def test_projected_gradient_rejects_general_rows() -> None:
    problem = QPProblem(
        Q=np.eye(2),
        c=np.zeros(2),
        A=np.array([[1.0, 1.0]]),
        b=np.array([1.0]),
        lb=np.zeros(2),
        ub=np.ones(2),
    )
    with pytest.raises(ValueError):
        solve_qp_projected_gradient(problem)
