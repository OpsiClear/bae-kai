# `bae-kai`: Bundle Adjustment in the Eager-mode

[![PyPI version](https://badge.fury.io/py/bae-kai.svg)](https://pypi.org/project/bae-kai/)
[![License](https://img.shields.io/badge/License-Apache%202.0%20%2B%20AGPL%203.0-blue.svg)](LICENSE)

> **⚠️ Development Phase Notice**: This library is in active development. APIs may change between releases.

`bae-kai` is a fork of [bae](https://github.com/zitongzhan/bae) with full Windows support, pre-built CUDA wheels, and cuDSS bundling. It provides PyTorch-based 2nd-order optimization for Bundle Adjustment (BA) and Pose Graph Optimization (PGO) using custom CUDA kernels and sparse block matrix operations.

## Features

- **Sparse Block Matrix Operations**: Optimized implementations of sparse matrix operations for large-scale optimization
- **CUDA Acceleration**: Custom CUDA kernels for high-performance sparse linear algebra
- **Bundle Adjustment**: Efficient implementation for camera pose and 3D structure optimization
- **Pose Graph Optimization**: Tools for optimizing robot trajectories using pose graph representations
- **PyTorch Integration**: Seamlessly integrates with PyTorch's automatic differentiation framework
- **Levenberg-Marquardt Optimizer**: Custom implementation of the LM algorithm for non-linear least squares problems

## Installation

### Prerequisites

- Python 3.12+
- PyTorch 2.0+ **with CUDA** (CPU-only PyTorch will not work)
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (for building from source)

### Step 1: Install PyTorch with CUDA

You must install a CUDA-enabled PyTorch **before** installing `bae-kai`. The CUDA version of PyTorch is not the default on PyPI, so you need to specify the index URL:

```bash
# CUDA 12.8 (recommended for RTX 30/40/50 series)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Or CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify your PyTorch has CUDA:

```bash
python -c "import torch; print(torch.version.cuda)"
# Should print something like "12.8", NOT "None"
```

### Step 2: Install bae-kai

`bae-kai` is distributed as a source package on PyPI. It compiles CUDA extensions during installation, which requires the CUDA Toolkit to be installed on your system.

```bash
pip install bae-kai --no-build-isolation
```

`--no-build-isolation` is required so the build can use your installed CUDA-enabled PyTorch.

**Windows**: The CUDA Toolkit is usually installed at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y` and is detected automatically.

**Linux**: Set `CUDA_HOME` if the toolkit is not at `/usr/local/cuda`:

```bash
CUDA_HOME=/usr/local/cuda-12.8 pip install bae-kai --no-build-isolation
```

### Pre-built Wheels

Pre-built wheels with bundled CUDA libraries are available as [GitHub Actions artifacts](https://github.com/OpsiClear/bae-kai/actions/workflows/build-wheels.yml) (no CUDA Toolkit needed to install):

| Platform | CUDA | Architectures |
|----------|------|---------------|
| Linux | 12.4, 12.8, 13.0 | sm_70 - sm_120 |
| Windows | 12.4, 12.6, 12.8 | sm_70 - sm_120 |

To install a pre-built wheel, download the `.whl` file for your platform and CUDA version from the latest successful workflow run, then:

```bash
pip install bae-0.1.2+cu12.8-cp312-cp312-win_amd64.whl
```

### From Source (Development)

```bash
git clone https://github.com/OpsiClear/bae-kai.git
cd bae-kai
uv sync
```

### Build Options

- `CUDA_HOME` / `CUDA_PATH`: Path to CUDA Toolkit (auto-detected on Windows)
- `BAE_SKIP_EXTENSIONS=1`: Skip CUDA extensions entirely (for sdist builds only)
- `USE_CUDSS`: `"1"` (default) to enable cuDSS support, `"0"` to disable
- `CUDSS_DIR`: Path to cuDSS installation if not in standard locations

## Usage

```python
from bae.optim import LM

# model: a torch.nn.Module whose forward() returns residuals
optimizer = LM(model, reject=30)

for idx in range(20):
    loss = optimizer.step(input)
    print(f'Iteration {idx}, loss: {loss.item()}')
```

The optimizer auto-selects the solver, damping strategy, and method. For explicit control:

```python
# String-based
optimizer = LM(model, solver="pcg", strategy="trustregion", method="schur")

# Object-based
from bae.utils import PCG, TrustRegion
optimizer = LM(model, solver=PCG(tol=1e-4, maxiter=250), strategy=TrustRegion())
```

See [`ba_example.py`](ba_example.py) for a complete Bundle Adjustment example using the BAL dataset.

### API Overview

| Module | Exports | Description |
|--------|---------|-------------|
| `bae.optim` | `LM`, `SchurLM` | Levenberg-Marquardt optimizer with auto-selection |
| `bae.autograd` | `TrackingTensor`, `map_transform`, `jacobian` | Sparse jacobian via operation tracing |
| `bae.utils` | `PCG`, `PCG_`, `CuDSS`, `SciPySpSolver` | Linear solvers |
| `bae.utils` | `TrustRegion`, `Adaptive` | Damping strategies |

### Integration with VGGT

`bae-kai` can be used as a Bundle Adjustment backend in [VGGT](https://github.com/zitongzhan/vggt) (Visual Geometry Grounded Transformer) to refine camera poses, intrinsics, and 3D points before exporting a COLMAP reconstruction:

```bash
python demo_colmap.py --scene_dir /path/to/scene --use_ba --implementation bae
```

## Citation

If you use `bae-kai` in your research, please cite the original paper:

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

Original code by Zitong Zhan et al. is licensed under [Apache 2.0](LICENSE). Additions in this fork are licensed under [AGPL 3.0](LICENSE-AGPL).
