import pytest

pytest.importorskip("highspy")

from opt_solver import (  # noqa: E402
    solve_lp_ipm,
    solve_lp_simplex,
    solve_qp_active_set,
    solve_qp_projected_gradient,
)
from opt_solver.benchmark import HighsReference  # noqa: E402
from opt_solver.problems import (  # noqa: E402
    afiro,
    beale,
    generate_lp,
    generate_qp,
    klee_minty,
)

LP_PROBLEMS = [
    beale(),
    klee_minty(3),
    afiro(),
    generate_lp(10, 6, seed=1),
    generate_lp(15, 9, seed=2),
    generate_lp(20, 12, seed=3),
    generate_lp(10, 6, seed=4, standard_form=True),
    generate_lp(20, 12, seed=5, standard_form=True),
]

QP_PROBLEMS = [
    generate_qp(5, 0, seed=1, box_only=True),
    generate_qp(10, 0, seed=2, box_only=True),
    generate_qp(20, 0, seed=3, box_only=True),
    generate_qp(10, 5, seed=4),
]


@pytest.mark.parametrize("problem", LP_PROBLEMS, ids=lambda problem: problem.name)
def test_lp_optimal_value_matches_highs(problem) -> None:
    reference = HighsReference().solve_lp(problem, method="simplex")
    simplex = solve_lp_simplex(problem, tol=1e-10)
    ipm = solve_lp_ipm(problem, tol=1e-10)

    assert reference["status"] == "optimal"
    assert simplex.status == "optimal"
    assert ipm.status == "optimal"
    assert abs(simplex.objective - float(reference["objective"])) <= 1e-8
    assert abs(ipm.objective - float(reference["objective"])) <= 1e-8


@pytest.mark.parametrize("problem", QP_PROBLEMS, ids=lambda problem: problem.name)
def test_qp_optimal_value_matches_highs(problem) -> None:
    reference = HighsReference().solve_qp(problem)
    active = solve_qp_active_set(problem)

    assert reference["status"] == "optimal"
    assert active.status == "optimal"
    assert abs(active.objective - float(reference["objective"])) <= 1e-8

    if problem.m == 0:
        projected = solve_qp_projected_gradient(problem, tol=1e-9)
        assert projected.status == "optimal"
        assert abs(projected.objective - float(reference["objective"])) <= 1e-8
