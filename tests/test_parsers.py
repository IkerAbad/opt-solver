from pathlib import Path

import numpy as np
import pytest

from opt_solver import read_lp, read_mps, solve_lp_simplex

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "classic"


def test_read_beale_lp() -> None:
    problem = read_lp(DATA_DIR / "beale.lp")

    assert problem.n == 4
    assert problem.m == 3
    assert problem.sense == "min"
    assert problem.c[0] == pytest.approx(-0.75)
    assert problem.c[1] == pytest.approx(150.0)
    assert problem.b == pytest.approx(np.array([0.0, 0.0, 1.0]))
    assert problem.row_sense == ["<=", "<=", "<="]
    assert problem.lb == pytest.approx(np.zeros(4))
    assert np.all(np.isinf(problem.ub))


def test_read_klee_minty_lp() -> None:
    problem = read_lp(DATA_DIR / "klee_minty_3.lp")

    assert problem.n == 3
    assert problem.m == 3
    assert problem.sense == "max"
    assert problem.c == pytest.approx(np.array([4.0, 2.0, 1.0]))
    assert problem.b == pytest.approx(np.array([5.0, 25.0, 25.0]))
    assert problem.row_sense == ["<=", "<=", "<="]


def test_read_simple_mps() -> None:
    problem = read_mps(DATA_DIR / "simple.mps")

    assert problem.n == 2
    assert problem.m == 2
    assert problem.c == pytest.approx(np.array([1.0, 2.0]))
    assert problem.b == pytest.approx(np.array([4.0, 2.0]))
    assert problem.row_sense == ["<=", ">="]
    assert problem.ub[0] == pytest.approx(3.0)
    assert problem.lb[1] == pytest.approx(0.0)

    solution = solve_lp_simplex(problem)
    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(4.0 / 3.0, abs=1e-9)


def test_read_afiro_mps() -> None:
    problem = read_mps(DATA_DIR / "afiro.mps")

    assert problem.n == 32
    assert problem.m == 27
    assert problem.sense == "min"

    solution = solve_lp_simplex(problem, tol=1e-10)
    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(-464.753142857143, abs=1e-6)


def test_lp_bounds_and_free_variable(tmp_path: Path) -> None:
    path = tmp_path / "bounds.lp"
    path.write_text(
        "\n".join(
            [
                "Minimize",
                " obj: x + y",
                "Subject To",
                " c1: x + y >= 1",
                "Bounds",
                " x free",
                " 0 <= y <= 2",
                "End",
            ]
        ),
        encoding="utf-8",
    )

    problem = read_lp(path)

    assert problem.n == 2
    assert problem.lb[0] == -np.inf
    assert problem.ub[0] == np.inf
    assert problem.lb[1] == pytest.approx(0.0)
    assert problem.ub[1] == pytest.approx(2.0)

    solution = solve_lp_simplex(problem)
    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(1.0)


def test_multiline_constraint(tmp_path: Path) -> None:
    path = tmp_path / "multiline.lp"
    path.write_text(
        "\n".join(
            [
                "Minimize",
                " obj: x + y",
                "Subject To",
                " c1: x + y",
                "     >= 1",
                "Bounds",
                " x >= 0",
                " y >= 0",
                "End",
            ]
        ),
        encoding="utf-8",
    )

    problem = read_lp(path)
    solution = solve_lp_simplex(problem)

    assert problem.m == 1
    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(1.0)


def test_integer_variables_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "integer.lp"
    path.write_text(
        "\n".join(
            [
                "Minimize",
                " obj: x",
                "Subject To",
                " c1: x >= 0",
                "General",
                " x",
                "End",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        read_lp(path)


def test_mps_ranges(tmp_path: Path) -> None:
    path = tmp_path / "ranges.mps"
    path.write_text(
        "\n".join(
            [
                "NAME          RANGETEST",
                "ROWS",
                " N  COST",
                " L  C1",
                "COLUMNS",
                "    X1        COST      1.0        C1        1.0",
                "RHS",
                "    RHS       C1        2.0",
                "RANGES",
                "    RNG       C1        3.0",
                "BOUNDS",
                " FR BND       X1",
                "ENDATA",
            ]
        ),
        encoding="utf-8",
    )

    problem = read_mps(path)
    solution = solve_lp_simplex(problem)

    assert problem.m == 2
    assert problem.row_sense == ["<=", "<="]
    assert problem.A[1, 0] == pytest.approx(-1.0)
    assert problem.b[1] == pytest.approx(1.0)
    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(-1.0)
