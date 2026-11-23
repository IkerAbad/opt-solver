# Infeasibility and unboundedness classification

The public interior-point solver, `solve_lp_ipm`, no longer assumes that the
problem is feasible and bounded. It runs two auxiliary phase I problems with
the same Mehrotra predictor-corrector before the main solve.

## Why phase I instead of the homogeneous self-dual embedding

The homogeneous self-dual (HSD) embedding is the standard way to obtain a
certificate of infeasibility or unboundedness. It is also a larger change: it
needs a skew-symmetric system with the extra variables `tau` and `kappa`, a
different initialization, a different step-length rule and a careful
interpretation of the limit points.

The phase I option reuses the existing, tested predictor-corrector. Both
auxiliary problems are feasible by construction and bounded below by zero, so
the main solver's assumptions hold for them. The classification is then read
directly from the optimal artificial sums. The cost is two extra interior-point
solves. This is the choice made here: it is smaller, easier to audit and
sufficient for the instances in the test suite.

The trade-off is explicit: this implementation does not return a Farkas
certificate. It returns a status and the phase I artificial sums. HSD would
return a certificate as well.

## Primal phase I

Given the standard form

```
minimize    c^T x
subject to  A x = b
            x >= 0
```

the primal phase I solves

```
minimize    sum(a)
subject to  A x + a = b
            x >= 0, a >= 0
```

After multiplying rows by `-1` when needed, `b >= 0`, so `x = 0`, `a = b` is a
feasible start. The objective is bounded below by zero. If the optimal
artificial sum is larger than the phase tolerance, the original primal is
infeasible.

## Dual phase I

The dual of the standard form is

```
maximize    b^T y
subject to  A^T y + s = c
            s >= 0
```

To test dual feasibility, the solver uses

```
minimize    sum(u)
subject to  A^T (y+ - y-) + s - u = c
            y+ >= 0, y- >= 0, s >= 0, u >= 0
```

The point `y+ = y- = 0`, `s = max(c, 0)`, `u = max(-c, 0)` is feasible and the
objective is bounded below by zero. If the optimal artificial sum is larger
than the phase tolerance, the dual is infeasible. Since the primal phase I has
already established primal feasibility, dual infeasibility means that the
primal is unbounded.

## Classification rule

Let `phase_tol = max(1e-8, 10 * tol)`. With the default `tol = 1e-10`, the phase
tolerance is `1e-8`.

1. Primal phase I artificial sum `> phase_tol`: return `infeasible`.
2. Primal phase I `<= phase_tol` and dual phase I `> phase_tol`: return
   `unbounded`.
3. Both phase I sums `<= phase_tol`: run the main predictor-corrector and
   return `optimal` or the main solver's status.
4. A phase I solve that does not converge and still has a positive artificial
   sum: return `numerical_error`. The solver does not guess.

For `status="infeasible"` or `status="unbounded"`, the returned vector is zero,
the objective is `NaN`, and `iterations` counts the phase I iterations that
were used. For `status="optimal"`, `iterations` is the sum of the phase I and
main iterations, while `history` records the main phase only.

## Results

The test suite in `tests/test_ipm_classification.py` compares the simplex
solver, the interior-point solver and HiGHS on the following instances. The
objective is the original objective, not the phase I artificial sum.

| Instance | Simplex | IPM | HiGHS | Objective | IPM iterations | Coincide |
|---|---|---|---|---|---:|---|
| infeasible_inequalities | infeasible | infeasible | infeasible | — | 7 | yes |
| infeasible_equalities | infeasible | infeasible | infeasible | — | 6 | yes |
| infeasible_redundant | infeasible | infeasible | infeasible | — | 7 | yes |
| unbounded_no_rows | unbounded | unbounded | unbounded | — | 6 | yes |
| unbounded_cone | unbounded | unbounded | unbounded | — | 13 | yes |
| unbounded_redundant | unbounded | unbounded | unbounded | — | 13 | yes |
| degenerate_equality | optimal | optimal | optimal | 1 | 18 | yes |
| degenerate_redundant_equality | optimal | optimal | optimal | 1 | 18 | yes |
| degenerate_redundant_inequality | optimal | optimal | optimal | 0 | 19 | yes |
| redundant_feasible | optimal | optimal | optimal | -1 | 19 | yes |
| redundant_consistent | optimal | optimal | optimal | 1 | 18 | yes |

The phase I artificial sums explain each classification:

| Instance | Primal phase I | Dual phase I | Classification |
|---|---:|---:|---|
| infeasible_inequalities | 1.00e+00 | — | infeasible |
| infeasible_equalities | 1.00e+00 | — | infeasible |
| infeasible_redundant | 1.00e+00 | — | infeasible |
| unbounded_no_rows | 0.00e+00 | 1.00e+00 | unbounded |
| unbounded_cone | 6.59e-11 | 2.00e+00 | unbounded |
| unbounded_redundant | 1.64e-10 | 1.00e+00 | unbounded |
| degenerate_equality | 1.89e-10 | 7.79e-11 | feasible |
| degenerate_redundant_equality | 1.48e-10 | 6.49e-11 | feasible |
| degenerate_redundant_inequality | 1.42e-10 | 1.28e-11 | feasible |
| redundant_feasible | 2.67e-10 | 4.32e-10 | feasible |
| redundant_consistent | 4.72e-11 | 6.67e-11 | feasible |

The degenerate cases have multiple optimal solutions or redundant active rows.
The redundant rows are consistent in the feasible cases and inconsistent in
`infeasible_redundant`. The phase I and the main solve both use `lstsq` as a
fallback when the reduced system is rank deficient.

## Limits

- The classifier is numerical. An instance that is infeasible by less than the
  phase tolerance can be accepted as feasible. The default tolerance is `1e-8`.
- No Farkas certificate is returned. The phase I artificial sums are diagnostic
  values, not certificates.
- If a phase I interior-point solve fails to converge, the result is
  `numerical_error`, not a guessed status.
- The dual phase I is larger than the original problem: it has `n` rows and
  `2m + 2n` variables. This is the main overhead visible in the benchmark.
- The raw `mehrotra` helper still assumes a feasible, bounded standard-form
  problem. The public `solve_lp_ipm` does not: it proves feasibility and
  boundedness before calling the helper.

## Reproduce

```bash
pytest -q tests/test_ipm_classification.py
pytest -q tests/test_infeasible_unbounded.py
```
