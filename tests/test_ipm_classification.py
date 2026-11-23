import numpy as np
import pytest

from opt_solver import LPProblem, solve_lp_ipm, solve_lp_simplex

CLASSIFICATION_CASES = [
    pytest.param(
        LPProblem(
            c=np.array([0.0]),
            A=np.array([[1.0], [-1.0]]),
            b=np.array([1.0, 0.0]),
            row_sense=[">=", ">="],
            lb=np.array([0.0]),
            ub=np.array([np.inf]),
        ),
        "infeasible",
        None,
        id="infeasible_inequalities",
    ),
    pytest.param(
        LPProblem(
            c=np.array([0.0]),
            A=np.array([[1.0], [1.0]]),
            b=np.array([1.0, 2.0]),
            row_sense=["=", "="],
            lb=np.array([0.0]),
            ub=np.array([np.inf]),
        ),
        "infeasible",
        None,
        id="infeasible_equalities",
    ),
    pytest.param(
        LPProblem(
            c=np.array([0.0, 0.0]),
            A=np.array([[1.0, 1.0], [2.0, 2.0]]),
            b=np.array([1.0, 3.0]),
            row_sense=["=", "="],
            lb=np.zeros(2),
            ub=np.full(2, np.inf),
        ),
        "infeasible",
        None,
        id="infeasible_redundant",
    ),
    pytest.param(
        LPProblem(
            c=np.array([-1.0]),
            A=np.zeros((0, 1)),
            b=np.zeros(0),
            row_sense=[],
            lb=np.array([0.0]),
            ub=np.array([np.inf]),
        ),
        "unbounded",
        None,
        id="unbounded_no_rows",
    ),
    pytest.param(
        LPProblem(
            c=np.array([-1.0, -1.0]),
            A=np.array([[1.0, -1.0]]),
            b=np.array([1.0]),
            row_sense=["<="],
            lb=np.zeros(2),
            ub=np.full(2, np.inf),
        ),
        "unbounded",
        None,
        id="unbounded_cone",
    ),
    pytest.param(
        LPProblem(
            c=np.array([-1.0, 0.0]),
            A=np.array([[1.0, -1.0], [2.0, -2.0]]),
            b=np.array([0.0, 0.0]),
            row_sense=["<=", "<="],
            lb=np.zeros(2),
            ub=np.full(2, np.inf),
        ),
        "unbounded",
        None,
        id="unbounded_redundant",
    ),
    pytest.param(
        LPProblem(
            c=np.array([1.0, 1.0]),
            A=np.array([[1.0, 1.0]]),
            b=np.array([1.0]),
            row_sense=["="],
            lb=np.zeros(2),
            ub=np.full(2, np.inf),
        ),
        "optimal",
        1.0,
        id="degenerate_equality",
    ),
    pytest.param(
        LPProblem(
            c=np.array([1.0, 1.0]),
            A=np.array([[1.0, 1.0], [1.0, 1.0]]),
            b=np.array([1.0, 1.0]),
            row_sense=["=", "="],
            lb=np.zeros(2),
            ub=np.full(2, np.inf),
        ),
        "optimal",
        1.0,
        id="degenerate_redundant_equality",
    ),
    pytest.param(
        LPProblem(
            c=np.array([0.0, 0.0]),
            A=np.array([[1.0, 1.0], [2.0, 2.0]]),
            b=np.array([1.0, 2.0]),
            row_sense=["<=", "<="],
            lb=np.zeros(2),
            ub=np.full(2, np.inf),
        ),
        "optimal",
        0.0,
        id="degenerate_redundant_inequality",
    ),
    pytest.param(
        LPProblem(
            c=np.array([-1.0, -1.0]),
            A=np.array([[1.0, 1.0], [1.0, 1.0]]),
            b=np.array([1.0, 1.0]),
            row_sense=["<=", "<="],
            lb=np.zeros(2),
            ub=np.full(2, np.inf),
        ),
        "optimal",
        -1.0,
        id="redundant_feasible",
    ),
    pytest.param(
        LPProblem(
            c=np.array([1.0, 1.0]),
            A=np.array([[1.0, 1.0], [2.0, 2.0]]),
            b=np.array([1.0, 2.0]),
            row_sense=["=", "="],
            lb=np.zeros(2),
            ub=np.full(2, np.inf),
        ),
        "optimal",
        1.0,
        id="redundant_consistent",
    ),
]


@pytest.mark.parametrize(
    "problem, expected_status, expected_objective",
    CLASSIFICATION_CASES,
)
def test_simplex_and_ipm_agree(
    problem: LPProblem,
    expected_status: str,
    expected_objective: float | None,
) -> None:
    simplex = solve_lp_simplex(problem, tol=1e-10)
    ipm = solve_lp_ipm(problem, tol=1e-10)

    assert simplex.status == expected_status
    assert ipm.status == expected_status

    if expected_objective is not None:
        assert simplex.objective == pytest.approx(expected_objective, abs=1e-8)
        assert ipm.objective == pytest.approx(expected_objective, abs=1e-8)
        assert "both phase i problems are feasible" in ipm.message.lower()
    else:
        assert ipm.iterations > 0
        assert "phase i" in ipm.message.lower()


@pytest.mark.parametrize(
    "problem, expected_status, expected_objective",
    CLASSIFICATION_CASES,
)
def test_ipm_matches_highs(
    problem: LPProblem,
    expected_status: str,
    expected_objective: float | None,
) -> None:
    pytest.importorskip("highspy")
    from opt_solver.benchmark import HighsReference

    reference = HighsReference().solve_lp(problem)
    ipm = solve_lp_ipm(problem, tol=1e-10)

    assert reference["status"] == expected_status
    assert ipm.status == expected_status
    if expected_objective is not None:
        assert abs(ipm.objective - float(reference["objective"])) <= 1e-8


def test_ipm_reports_primal_phase_one_for_infeasible() -> None:
    problem = LPProblem(
        c=np.array([0.0]),
        A=np.array([[1.0], [-1.0]]),
        b=np.array([1.0, 0.0]),
        row_sense=[">=", ">="],
        lb=np.array([0.0]),
        ub=np.array([np.inf]),
    )
    solution = solve_lp_ipm(problem)

    assert solution.status == "infeasible"
    assert "primal phase i is positive" in solution.message.lower()


def test_ipm_reports_dual_phase_one_for_unbounded() -> None:
    problem = LPProblem(
        c=np.array([-1.0, -1.0]),
        A=np.array([[1.0, -1.0]]),
        b=np.array([1.0]),
        row_sense=["<="],
        lb=np.zeros(2),
        ub=np.full(2, np.inf),
    )
    solution = solve_lp_ipm(problem)

    assert solution.status == "unbounded"
    assert "dual phase i is positive" in solution.message.lower()
    assert "unbounded" in solution.message.lower()
