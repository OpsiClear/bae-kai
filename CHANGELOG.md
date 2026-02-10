# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-02-10

### Fixed
- sdist now includes C++/CUDA source files (MANIFEST.in)
- setup.py respects BAE_SKIP_EXTENSIONS for sdist builds
- PyPI package name corrected to bae-kai
- CI: Linux builds use bare runner instead of Docker containers

## [0.1.1] - 2026-02-09

### Added
- Auto-selection API for LM optimizer (solver, strategy, method)
- String-based configuration (`solver="pcg"`, `method="schur"`)
- SchurLM refactored as thin LM wrapper
- Public exports in `bae.autograd` and `bae.utils` `__init__` files
- Full Windows CUDA extension support
- Blackwell (sm_120) architecture support for RTX 50-series
- CUDA 12.4, 12.6, 12.8, 13.0 wheel builds
- cuDSS bundling in wheels for self-contained installation
- Comprehensive test coverage
- Performance optimizations (verbose flag, contiguous guards)
- GitHub Actions CI/CD workflows
- PyPI publishing workflow

### Changed
- Dead code and unused imports removed from autograd, sparse modules
- Dual license: original code Apache-2.0, additions AGPL-3.0-only
- CUDA extensions now required (no Python fallbacks)

### Fixed
- `_compute_jd` shape bug with list-mode jacobians in strategy update
- Disk space issues in CI builds
- MSVC linker detection on Windows
- Triton compatibility on Windows

## [0.1.0] - 2025-01-01

### Added
- Initial release
- Bundle Adjustment in Eager mode
- Sparse Jacobian computation
- Levenberg-Marquardt optimizer
- PCG and CuDSS solvers
- BAL dataset loader
