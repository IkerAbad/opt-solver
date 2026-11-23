import numpy as np

from opt_solver import LPProblem, solve_lp_ipm, solve_lp_simplex


def test_infeasible_inequalities() -> None:
    problem = LPProblem(
        c=np.array([0.0]),
        A=np.array([[1.0], [-1.0]]),
        b=np.array([1.0, 0.0]),
        row_sense=[">=", ">="],
        lb=np.array([0.0]),
        ub=np.array([np.inf]),
    )
    simplex = solve_lp_simplex(problem)
    ipm = solve_lp_ipm(problem)

    assert simplex.status == "infeasible"
    assert ipm.status == "infeasible"


def test_infeasible_equalities() -> None:
    problem = LPProblem(
        c=np.array([0.0]),
        A=np.array([[1.0], [1.0]]),
        b=np.array([1.0, 2.0]),
        row_sense=["=", "="],
        lb=np.array([0.0]),
        ub=np.array([np.inf]),
    )
    simplex = solve_lp_simplex(problem)
    ipm = solve_lp_ipm(problem)

    assert simplex.status == "infeasible"
    assert ipm.status == "infeasible"


def test_unbounded_without_rows() -> None:
    problem = LPProblem(
        c=np.array([-1.0]),
        A=np.zeros((0, 1)),
        b=np.zeros(0),
        row_sense=[],
        lb=np.array([0.0]),
        ub=np.array([np.inf]),
    )
    simplex = solve_lp_simplex(problem)
    ipm = solve_lp_ipm(problem)

    assert simplex.status == "unbounded"
    assert ipm.status == "unbounded"


def test_unbounded_inside_a_cone() -> None:
    problem = LPProblem(
        c=np.array([-1.0, -1.0]),
        A=np.array([[1.0, -1.0]]),
        b=np.array([1.0]),
        row_sense=["<="],
        lb=np.zeros(2),
        ub=np.full(2, np.inf),
    )
    simplex = solve_lp_simplex(problem)
    ipm = solve_lp_ipm(problem)

    assert simplex.status == "unbounded"
    assert ipm.status == "unbounded"
