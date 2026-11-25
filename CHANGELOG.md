# Changelog

All notable changes to this project are documented in this file. The format
follows Keep a Changelog and the project uses semantic versioning.

## [0.1.1] - 2026-11-25

### Changed

- Distribution name changed to `optkit`. The previous name carried a personal
  prefix only because `opt-solver` was taken on PyPI. The import package is
  still `opt_solver` and the repository is still `opt-solver`.

## [0.1.0] - 2026-11-23

- Two-phase simplex with Bland's rule, Mehrotra interior point with phase I
  classification, active-set QP and accelerated projected gradient.
- MPS and CPLEX LP readers.
- 36 comparisons against HiGHS.
