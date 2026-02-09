# `bae-kai`: Bundle Adjustment in the Eager-mode

[![PyPI version](https://badge.fury.io/py/bae-kai.svg)](https://pypi.org/project/bae-kai/)
[![License](https://img.shields.io/badge/License-Apache%202.0%20%2B%20AGPL%203.0-blue.svg)](LICENSE)

> **⚠️ Development Phase Notice**: This library is currently in active development. APIs are subject to change and should be considered experimental. Use at your own discretion in production environments.

`bae-kai` is a fork of [bae](https://github.com/zitongzhan/bae) with full Windows support, pre-built CUDA wheels, and cuDSS bundling.

`bae` is a PyTorch-based library supporting 2nd-order optimization techniques. The library provides efficient implementations for sparse optimization problems in robotics, particularly Bundle Adjustment (BA) and Pose Graph Optimization (PGO).

## News

- 2026-02-06: Windows CUDA wheels now available on PyPI
- 2025-12-12: Added a VGGT integration example.

## Features

- **Sparse Block Matrix Operations**: Optimized implementations of sparse matrix operations for large-scale optimization
- **CUDA Acceleration**: Custom CUDA kernels for high-performance sparse linear algebra
- **Bundle Adjustment**: Efficient implementation for camera pose and 3D structure optimization
- **Pose Graph Optimization**: Tools for optimizing robot trajectories using pose graph representations
- **PyTorch Integration**: Seamlessly integrates with PyTorch's automatic differentiation framework
- **Levenberg-Marquardt Optimizer**: Custom implementation of the LM algorithm for non-linear least squares problems

### Future Plan
- [ ] An new backend for [distributed solver](https://github.com/NVIDIA/AMGX)  

## Installation

### Quick Install (Recommended)

Pre-built wheels with CUDA extensions are available on PyPI:

```bash
# Install with CUDA 12.8 (for RTX 30/40/50 series)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install bae-kai

# Or with CUDA 13.0 (latest)
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install bae-kai
```

### Available Wheels

| Platform | CUDA | Architectures |
|----------|------|---------------|
| Linux | 12.4, 12.6, 12.8, 13.0 | sm_70 - sm_120 |
| Windows | 12.4, 12.6, 12.8 | sm_70 - sm_120 |

### From Source

For development or custom builds:

```bash
# Clone this repository
git clone https://github.com/OpsiClear/bae-kai.git
cd bae-kai

# Install PyPose from the bae branch
uv pip install git+https://github.com/pypose/pypose.git@bae

# Install in development mode (uv sync handles all dependencies)
uv sync
```

### Build Options

Control the build with environment variables:

- `BAE_BUILD_EXTENSIONS=1`: Force building CUDA extensions (required on Windows)
- `BAE_SKIP_EXTENSIONS=1`: Skip CUDA extensions (Python fallbacks not available)
- `USE_CUDSS`: Set to "1" (default) to enable cuDSS support, "0" to disable
- `CUDSS_DIR`: Path to cuDSS installation if not in standard locations

## Example Usage

### Bundle Adjustment

Bundle Adjustment optimizes camera poses and 3D point positions to minimize reprojection error. The following example shows how to perform BA using `bae`:

```python
import torch
from datapipes.bal_loader import get_problem
from ba_helpers import Reproj
from bae.optim import LM

# Load a problem from the BAL dataset
dataset = get_problem("problem-49-7776-pre", "ladybug", use_quat=True)
dataset = {k: v.to('cuda') for k, v in dataset.items() if isinstance(v, torch.Tensor)}

# Prepare input for the optimization
input = {
    "points_2d": dataset['points_2d'],
    "camera_indices": dataset['camera_index_of_observations'],
    "point_indices": dataset['point_index_of_observations']
}

# Initialize model with camera parameters and 3D points
model = Reproj(
    dataset['camera_params'].clone(),
    dataset['points_3d'].clone()
).to('cuda')

# Auto-selection: solver, strategy, and method are chosen automatically
optimizer = LM(model, reject=30)

# Run optimization for multiple iterations
for idx in range(20):
    loss = optimizer.step(input)
    print(f'Iteration {idx}, loss: {loss.item()}')
```

For more control, you can configure the optimizer explicitly:

```python
# String-based configuration
optimizer = LM(model, solver="pcg", strategy="trustregion", method="schur")

# Object-based configuration (backward-compatible)
from bae.utils.pysolvers import PCG
from bae.utils.schur import TrustRegion
optimizer = LM(model, solver=PCG(tol=1e-4, maxiter=250), strategy=TrustRegion())
```

See [`ba_example.py`](ba_example.py) for a complete working example.

### API Overview

| Module | Exports | Description |
|--------|---------|-------------|
| `bae.optim` | `LM`, `SchurLM` | Levenberg-Marquardt optimizer with auto-selection |
| `bae.autograd` | `TrackingTensor`, `map_transform`, `jacobian` | Sparse jacobian via operation tracing |
| `bae.utils` | `PCG`, `PCG_`, `CuDSS`, `SciPySpSolver` | Linear solvers |
| `bae.utils` | `TrustRegion`, `Adaptive` | Damping strategies |

### Integration with VGGT

`bae` is used as an optional Bundle Adjustment backend in [our VGGT fork](https://github.com/zitongzhan/vggt) (Visual Geometry Grounded Transformer) to refine the camera poses, intrinsics, and 3D points predicted by VGGT before exporting a COLMAP reconstruction.

After installing `bae`, you can run VGGT's COLMAP export with BA enabled and `bae` selected as the solver:

```bash
python demo_colmap.py --scene_dir /path/to/scene --use_ba --implementation bae  # optional: --shared_camera
```

This command invokes `prepare_bae(...)` inside `vggt/demo_colmap.py`, which wraps VGGT tracks and predictions into `bae.optim.LM` and updates `extrinsic`, `intrinsic`, and `points_3d` in place before writing `scene_dir/sparse/` in COLMAP format.

## Dataset Support

The library supports common optimization datasets and tasks:

- **Bundle Adjustment in the Large (BAL)** dataset
- **1DSfM** dataset for large-scale structure from motion
- **G2O** pose graph datasets

## Performance

`bae` is designed for high performance using:

- Efficient sparse block matrix operations
- CUDA acceleration for core operations
- Optimized linear solvers (PCG, CUDA Sparse Solver)
- Memory-efficient data structures

## Citation

If you use `bae` in your research, please cite:

```bibtex
@article{zhan2025bundle,
  title = {Bundle Adjustment in the Eager Mode},
  author = {Zhan, Zitong and Xu, Huan and Fang, Zihang and Wei, Xinpeng and Hu, Yaoyu and Wang, Chen},
  journal = {arXiv preprint arXiv:2409.12190},
  year = {2025},
  url = {https://arxiv.org/abs/2409.12190}
}
```

## Acknowledgements

This project is a fork of [bae](https://github.com/zitongzhan/bae) by Zitong Zhan et al.

The implementation draws inspiration from:
- [bae (original)](https://github.com/zitongzhan/bae) - Bundle Adjustment in the Eager Mode
- [PyPose](https://github.com/pypose/pypose) for SE(3) pose representations
- GTSAM for reprojection jacobian concepts

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE-AGPL).
