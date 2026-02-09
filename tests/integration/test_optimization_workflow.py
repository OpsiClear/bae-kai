"""Integration tests for optimization workflows.

Tests core components work together correctly.
"""

import pytest
import torch
import torch.nn as nn

from bae.autograd.function import TrackingTensor, map_transform
from bae.utils.pysolvers import PCG, SciPySpSolver


class TestTrackingTensorIntegration:
    """Test TrackingTensor works correctly."""

    def test_tracking_tensor_is_subclass(self):
        """Test TrackingTensor is a proper tensor subclass."""
        n = 5
        data = torch.randn(n, dtype=torch.float64)
        tracked = TrackingTensor(data)

        assert isinstance(tracked, TrackingTensor)
        assert isinstance(tracked, torch.Tensor)
        assert tracked.shape == data.shape

    def test_tracking_tensor_indexing(self):
        """Test TrackingTensor supports indexing."""
        data = torch.randn(10, 3, dtype=torch.float64)
        tracked = TrackingTensor(data)

        indices = torch.tensor([0, 2, 5])
        result = tracked[indices]

        assert result.shape == (3, 3)
        assert hasattr(result, 'optrace')

    def test_map_transform_creates_optrace(self):
        """Test @map_transform creates operation trace."""
        @map_transform
        def simple_fn(x):
            return x * 2

        data = torch.randn(5, 3, dtype=torch.float64)
        tracked = TrackingTensor(data)

        result = simple_fn(tracked)

        assert hasattr(result, 'optrace')
        assert id(result) in result.optrace


class TestSolverIntegration:
    """Test solvers work with sparse matrices."""

    def _make_spd_system(self, n, dtype, device):
        """Create symmetric positive definite system."""
        torch.manual_seed(42)
        M = torch.randn(n + 5, n, dtype=dtype, device=device)
        A = M.T @ M + 0.1 * torch.eye(n, dtype=dtype, device=device)
        b = torch.randn(n, dtype=dtype, device=device)
        return A, b

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_pcg_dense_solves_correctly(self, device):
        """Test PCG solves dense systems."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        n = 10
        A, b = self._make_spd_system(n, torch.float64, device)

        solver = PCG(tol=1e-8, maxiter=100)
        x = solver(A, b)

        residual = torch.linalg.norm(A @ x - b) / torch.linalg.norm(b)
        assert residual < 1e-6

    def test_scipy_solver_works(self):
        """Test SciPy solver on CPU."""
        n = 10
        A, b = self._make_spd_system(n, torch.float64, "cpu")
        A_sparse = A.to_sparse_csr()

        solver = SciPySpSolver()
        x = solver(A_sparse, b)

        residual = torch.linalg.norm(A @ x - b) / torch.linalg.norm(b)
        assert residual < 1e-8


class TestExceptionIntegration:
    """Test exception handling in optimization context."""

    def test_bae_exceptions_importable(self):
        """Test custom exceptions can be imported."""
        from bae.exceptions import (
            BAEError,
            BAESparsityError,
            BAEOptimizationError,
            BAESolverError,
        )

        # Verify they're proper exception classes
        assert issubclass(BAESparsityError, BAEError)
        assert issubclass(BAESolverError, BAEOptimizationError)


class TestTrustRegionIntegration:
    """Test TrustRegion strategy with different input formats."""

    def test_trust_region_with_single_tensor(self):
        """Test TrustRegion accepts single CSR jacobian tensor."""
        from bae.utils.schur import TrustRegion

        # Create mock parameter group
        pg = {
            'damping': 1e-4,
            'radius': 1e4,
            'high': 0.75,
            'low': 0.25,
            'up': 2.0,
            'down': 2.0,
            'factor': 2.0,
        }

        # Create single CSR jacobian (simulating optimizer's output)
        torch.manual_seed(42)
        m, n = 100, 20
        J_dense = torch.randn(m, n, dtype=torch.float64)
        J_csr = J_dense.to_sparse_csr()

        # Update vector (full)
        D = torch.randn(n, 1, dtype=torch.float64)

        # Residual
        R = torch.randn(m, 1, dtype=torch.float64)

        # Should not raise
        strategy = TrustRegion()
        strategy.update(pg, last=100.0, loss=90.0, J=J_csr, D=D, R=R)

        # Damping should be updated
        assert 'damping' in pg
        assert 'radius' in pg

    def test_trust_region_with_jacobian_list(self):
        """Test TrustRegion accepts list of BSR jacobians."""
        from bae.utils.schur import TrustRegion

        # Create mock parameter group
        pg = {
            'damping': 1e-4,
            'radius': 1e4,
            'high': 0.75,
            'low': 0.25,
            'up': 2.0,
            'down': 2.0,
            'factor': 2.0,
        }

        torch.manual_seed(42)

        # Create list of jacobians (simulating original format)
        J_list = []
        D_list = []
        m = 100
        for n in [10, 8]:
            J_dense = torch.randn(m, n, dtype=torch.float64)
            # Create simple BSR-like structure
            J_list.append(J_dense)
            D_list.append(torch.randn(n, 1, dtype=torch.float64))

        # Residual
        R = torch.randn(m, 1, dtype=torch.float64)

        # Should not raise
        strategy = TrustRegion()
        strategy.update(pg, last=100.0, loss=90.0, J=J_list, D=D_list, R=R)

        # Damping should be updated
        assert 'damping' in pg

    def test_adaptive_with_single_tensor(self):
        """Test Adaptive strategy accepts single CSR jacobian tensor."""
        from bae.utils.schur import Adaptive

        # Create mock parameter group
        pg = {
            'damping': 1e-4,
            'high': 0.75,
            'low': 0.25,
            'up': 2.0,
            'down': 0.5,
        }

        torch.manual_seed(42)
        m, n = 100, 20
        J_dense = torch.randn(m, n, dtype=torch.float64)
        J_csr = J_dense.to_sparse_csr()

        D = torch.randn(n, 1, dtype=torch.float64)
        R = torch.randn(m, 1, dtype=torch.float64)

        # Should not raise
        strategy = Adaptive()
        strategy.update(pg, last=100.0, loss=90.0, J=J_csr, D=D, R=R)

        assert 'damping' in pg
