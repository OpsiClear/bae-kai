# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`bae` (Bundle Adjustment in the Eager-mode) is a PyTorch library for 2nd-order sparse optimization, focused on Bundle Adjustment (BA) and Pose Graph Optimization (PGO). It uses custom CUDA kernels and sparse block matrix operations for large-scale computer vision problems.

**Status**: Active development (v0.1), APIs are experimental.

## Build Commands

### Linux
```bash
# Sync dependencies and install package with CUDA extensions
uv sync

# Disable CUDSS if not available
USE_CUDSS=0 uv sync
```

### Windows

Windows builds work automatically with the MSVC linker fix in setup.py. The setup.py:
- Automatically sets `DISTUTILS_USE_SDK=1`
- Copies MSVC's link.exe to `.msvc_bin/` and prepends it to PATH (avoiding conflict with Git's link.exe)
- Defines `USE_CUDA` to work around PyTorch cu128 namespace collision
- Auto-detects CUDSS at `C:\Program Files\NVIDIA cuDSS\v0.7`

```cmd
# From any terminal (no need for VS Developer prompt)
uv pip install --no-build-isolation -e .

# Or using uv sync
uv sync
```

**Features on Windows**:
- **Triton**: Supported via `triton-windows` package (auto-installed)
- **CUDSS**: Supported if installed at standard path or via `CUDSS_DIR` environment variable

## Testing

```bash
# Run graph jacobian tests
uv run pytest tests/autograd/test_graph_jacobian.py

# Run all autograd tests
uv run pytest tests/autograd/
```

## Running Examples

```bash
# Bundle Adjustment example (requires BAL dataset download)
uv run python ba_example.py

# Pose Graph Optimization
uv run python pgo.py
```

## Architecture

### Core Modules (`bae/`)

- **`autograd/`**: Custom jacobian computation through operation tracing
  - `TrackingTensor`: Subclass of `torch.Tensor` that records operations (`index` or `map`) in an `optrace` dict
  - `jacobian()`: Walks the operation trace backward to compute sparse block jacobians via `torch.vmap(jacrev())`
  - `@map_transform`: Decorator to register vectorized functions for jacobian tracking

- **`sparse/`**: Sparse block matrix operations with multiple backends
  - BSR (Block Sparse Row) format for efficient matrix operations
  - CUDA kernels: cuSparse wrapper, custom kernels, optional CUDSS solver
  - `py_ops.py`: Python operations including Triton kernels

- **`optim/`**: Custom Levenberg-Marquardt optimizer extending PyPose's implementation
  - Integrates with the autograd jacobian system
  - Uses sparse JᵀJ computation and configurable linear solvers

- **`utils/`**: Linear solvers (PCG, CuDSS), Schur complement utilities

### Key Patterns

1. **TrackingTensor flow**: Parameters are wrapped in `TrackingTensor`, operations are traced, then `jacobian(output, params)` computes sparse jacobians by walking the trace backward.

2. **SE(3) handling**: Parameters with `trim_SE3_grad=True` attribute use 6-DOF tangent space updates while storing 7-element quaternion+translation.

3. **Solver selection**: `PCG(tol, maxiter)` for CPU/GPU iterative solving, `CuDSS()` for direct GPU solving (requires CUDSS library).

### Data Loading (`datapipes/`)

- `bal_loader.py`: Loads BAL (Bundle Adjustment in the Large) datasets
- Supports BAL, 1DSfM, and G2O formats

## Environment Variables

- `USE_CUDSS`: "1" (default) to enable CUDSS support, "0" to disable
- `CUDSS_DIR`: Path to CUDSS installation if not in standard location
  - Windows default: `C:\Program Files\NVIDIA cuDSS\v0.7`
  - Linux default: `/usr/include/libcudss/12/`

## Dependencies

- PyTorch 2.0+ with CUDA 12.8 (configured via uv index)
- PyPose (bae branch) for SE(3)/se(3) Lie group operations
- scipy, torchdata, warp-lang
