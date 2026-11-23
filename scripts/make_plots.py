#!/usr/bin/env python3
"""Generate convergence, sensitivity and scaling figures."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from opt_solver import (
    LPProblem,
    QPProblem,
    solve_lp_ipm,
    solve_lp_simplex,
    solve_qp_active_set,
)
from opt_solver.problems import afiro, generate_lp

FIGURES = Path("docs/figures")


def plot_ipm_convergence() -> None:
    problem = afiro()
    solution = solve_lp_ipm(problem, tol=1e-10, max_iter=100, store_history=True)
    history = solution.history
    iterations = [entry["iteration"] for entry in history]

    def positive(values: list[float]) -> list[float]:
        return [max(value, 1e-16) for value in values]

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.semilogy(
        iterations,
        positive([entry["mu"] for entry in history]),
        label="complementarity μ",
        color="tab:blue",
    )
    axis.semilogy(
        iterations,
        positive([entry["primal_residual"] for entry in history]),
        label="primal residual",
        color="tab:orange",
    )
    axis.semilogy(
        iterations,
        positive([entry["dual_residual"] for entry in history]),
        label="dual residual",
        color="tab:green",
    )
    axis.set_xlabel("iteration")
    axis.set_ylabel("value")
    axis.set_title("Mehrotra predictor-corrector on AFIRO")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES / "ipm_convergence.png", dpi=160)
    plt.close(figure)


def plot_sensitivity() -> None:
    base_lp = LPProblem(
        c=np.array([2.0, 2.0]),
        A=np.array([[1.0, 1.0]]),
        b=np.array([4.0]),
        row_sense=["<="],
        lb=np.zeros(2),
        ub=np.array([2.0, 3.0]),
        sense="max",
        name="sensitivity_lp",
    )

    coefficient_values = np.linspace(0.0, 4.0, 81)
    objective_values: list[float] = []
    x_values: list[float] = []
    for coefficient in coefficient_values:
        solution = solve_lp_simplex(
            replace(base_lp, c=np.array([coefficient, 2.0])),
            tol=1e-10,
        )
        if solution.status == "optimal":
            objective_values.append(solution.objective)
            x_values.append(float(solution.x[0]))
        else:
            objective_values.append(float("nan"))
            x_values.append(float("nan"))

    base_qp = QPProblem(
        Q=np.array([[1.0]]),
        c=np.array([0.0]),
        A=np.zeros((0, 1)),
        b=np.zeros(0),
        lb=np.array([-1.0]),
        ub=np.array([1.0]),
        name="sensitivity_qp",
    )
    qp_coefficients = np.linspace(-5.0, 5.0, 101)
    qp_x_values: list[float] = []
    qp_objective_values: list[float] = []
    for coefficient in qp_coefficients:
        solution = solve_qp_active_set(
            replace(base_qp, c=np.array([coefficient])),
            tol=1e-10,
        )
        qp_x_values.append(float(solution.x[0]))
        qp_objective_values.append(solution.objective)

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))

    axes[0, 0].plot(coefficient_values, objective_values, color="tab:blue")
    axes[0, 0].set_title("LP: optimal value vs objective coefficient")
    axes[0, 0].set_xlabel("c1")
    axes[0, 0].set_ylabel("f*")

    axes[0, 1].plot(coefficient_values, x_values, color="tab:orange")
    axes[0, 1].set_title("LP: optimal x1 vs objective coefficient")
    axes[0, 1].set_xlabel("c1")
    axes[0, 1].set_ylabel("x1*")

    axes[1, 0].plot(qp_coefficients, qp_x_values, color="tab:green")
    axes[1, 0].set_title("QP: optimal x vs linear coefficient")
    axes[1, 0].set_xlabel("c")
    axes[1, 0].set_ylabel("x*")

    axes[1, 1].plot(qp_coefficients, qp_objective_values, color="tab:red")
    axes[1, 1].set_title("QP: optimal value vs linear coefficient")
    axes[1, 1].set_xlabel("c")
    axes[1, 1].set_ylabel("f*")

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(FIGURES / "sensitivity.png", dpi=160)
    plt.close(figure)


def plot_scaling() -> None:
    sizes = [20, 40, 80, 160]
    simplex_iterations: list[int] = []
    ipm_iterations: list[int] = []
    simplex_times: list[float] = []
    ipm_times: list[float] = []
    simplex_statuses: list[str] = []

    for n in sizes:
        problem = generate_lp(n, max(1, n // 2), seed=n)

        start = time.perf_counter()
        simplex = solve_lp_simplex(problem, tol=1e-10)
        simplex_times.append(time.perf_counter() - start)
        simplex_iterations.append(simplex.iterations)
        simplex_statuses.append(simplex.status)

        start = time.perf_counter()
        ipm = solve_lp_ipm(problem, tol=1e-10)
        ipm_times.append(time.perf_counter() - start)
        ipm_iterations.append(ipm.iterations)

        print(
            f"n={n:4d} simplex={simplex.status:8s} {simplex.iterations:5d} it "
            f"{simplex_times[-1]:.4f} s | ipm={ipm.status:8s} {ipm.iterations:3d} it "
            f"{ipm_times[-1]:.4f} s"
        )

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    axes[0].plot(sizes, simplex_iterations, marker="o", label="simplex")
    axes[0].plot(sizes, ipm_iterations, marker="s", label="interior point")
    axes[0].set_title("Iterations on generated box LPs")
    axes[0].set_xlabel("variables n")
    axes[0].set_ylabel("iterations")
    axes[0].set_xticks(sizes)
    if any(status != "optimal" for status in simplex_statuses):
        axes[0].text(
            0.03,
            0.95,
            "simplex reaches the iteration limit",
            transform=axes[0].transAxes,
            fontsize=8,
            va="top",
        )

    axes[1].plot(sizes, simplex_times, marker="o", label="simplex")
    axes[1].plot(sizes, ipm_times, marker="s", label="interior point")
    axes[1].set_title("Time on generated box LPs")
    axes[1].set_xlabel("variables n")
    axes[1].set_ylabel("seconds")
    axes[1].set_xticks(sizes)

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.tight_layout()
    figure.savefig(FIGURES / "scaling.png", dpi=160)
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_ipm_convergence()
    plot_sensitivity()
    plot_scaling()
    print(f"Figures written to {FIGURES.resolve()}")


if __name__ == "__main__":
    main()
