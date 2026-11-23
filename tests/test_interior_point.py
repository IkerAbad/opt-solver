import pytest

from opt_solver import solve_lp_ipm, solve_lp_simplex
from opt_solver.problems import afiro, beale, generate_lp


@pytest.mark.parametrize(
    "problem",
    [
        beale(),
        afiro(),
        generate_lp(10, 6, seed=1),
        generate_lp(10, 6, seed=2, standard_form=True),
        generate_lp(20, 12, seed=3),
    ],
    ids=lambda problem: problem.name,
)
def test_ipm_matches_simplex(problem) -> None:
    simplex = solve_lp_simplex(problem, tol=1e-10)
    ipm = solve_lp_ipm(problem, tol=1e-10, store_history=True)

    assert simplex.status == "optimal"
    assert ipm.status == "optimal"
    assert ipm.objective == pytest.approx(simplex.objective, abs=1e-8)
    assert len(ipm.history) > 0
    assert ipm.history[-1]["mu"] <= 1e-9
