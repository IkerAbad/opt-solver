#!/usr/bin/env python3
"""Run the LP and QP benchmarks against HiGHS."""

from __future__ import annotations

import argparse
from pathlib import Path

from opt_solver.benchmark import (
    benchmark_lp,
    benchmark_qp,
    format_markdown,
    format_summary,
    write_csv,
)
from opt_solver.problems import afiro, beale, generate_lp, generate_qp, klee_minty
from opt_solver.types import LPProblem, QPProblem


def build_problems(quick: bool) -> tuple[list[LPProblem], list[QPProblem]]:
    lp_problems: list[LPProblem] = [beale(), klee_minty(3), afiro()]
    if not quick:
        lp_problems.append(klee_minty(4))

    lp_sizes = [10, 20] if quick else [10, 20, 40, 80]
    for n in lp_sizes:
        m = max(1, n // 2)
        lp_problems.append(generate_lp(n, m, seed=n))
        lp_problems.append(generate_lp(n, m, seed=1000 + n, standard_form=True))

    qp_sizes = [5, 10] if quick else [5, 10, 20, 40]
    qp_problems: list[QPProblem] = [
        generate_qp(n, 0, seed=n, box_only=True) for n in qp_sizes
    ]
    qp_problems.extend(generate_qp(n, max(1, n // 2), seed=2000 + n) for n in qp_sizes)

    return lp_problems, qp_problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a small subset.")
    parser.add_argument("--output", default="results/benchmark.csv", help="CSV output path.")
    parser.add_argument(
        "--markdown",
        default="results/benchmark.md",
        help="Markdown table output path.",
    )
    args = parser.parse_args()

    lp_problems, qp_problems = build_problems(args.quick)
    rows = benchmark_lp(lp_problems) + benchmark_qp(qp_problems)

    print(format_summary(rows))
    write_csv(rows, args.output)

    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(format_markdown(rows) + "\n", encoding="utf-8")

    failures = sum(1 for row in rows if not bool(row["within_tolerance"]))
    print(f"\n{len(rows)} solver runs, {failures} outside the 1e-8 objective tolerance.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
