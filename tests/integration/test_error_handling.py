"""Integration tests for error handling.

Tests that custom exceptions are raised appropriately and provide
useful error messages.
"""

import pytest
import torch

from bae.exceptions import (
    BAEError,
    BAESparsityError,
    BAEOptimizationError,
    BAESolverError,
    BAEDeviceError,
)
from bae.sparse.py_ops import diagonal_op_


class TestCustomExceptions:
    """Test custom exception hierarchy."""

    def test_exception_hierarchy(self):
        """Test exception inheritance."""
        assert issubclass(BAESparsityError, BAEError)
        assert issubclass(BAEOptimizationError, BAEError)
        assert issubclass(BAESolverError, BAEOptimizationError)
        assert issubclass(BAEDeviceError, BAEError)

    def test_exception_messages(self):
        """Test exceptions can be raised with messages."""
        with pytest.raises(BAESparsityError) as exc_info:
            raise BAESparsityError("Test sparsity error")
        assert "sparsity" in str(exc_info.value).lower()

        with pytest.raises(BAESolverError) as exc_info:
            raise BAESolverError("Solver failed to converge")
        assert "Solver" in str(exc_info.value)


class TestSparsityErrors:
    """Test sparsity-related error handling."""

    def test_diagonal_op_non_square_blocks(self):
        """Test diagonal_op_ raises error for non-square blocks."""
        # Create BSR tensor with non-square blocks (2x3)
        crow_indices = torch.tensor([0, 1])
        col_indices = torch.tensor([0])
        values = torch.randn(1, 2, 3)  # Non-square blocks

        bsr = torch.sparse_bsr_tensor(
            crow_indices, col_indices, values,
            size=(2, 3)
        )

        with pytest.raises(BAESparsityError) as exc_info:
            diagonal_op_(bsr)

        error_msg = str(exc_info.value)
        assert "square" in error_msg.lower()
        assert "2" in error_msg and "3" in error_msg

    def test_diagonal_op_nonzero_offset(self):
        """Test diagonal_op_ raises error for non-zero offset."""
        # Create BSR tensor with square blocks
        crow_indices = torch.tensor([0, 1, 2])
        col_indices = torch.tensor([0, 1])
        values = torch.randn(2, 2, 2)

        bsr = torch.sparse_bsr_tensor(
            crow_indices, col_indices, values,
            size=(4, 4)
        )

        with pytest.raises(BAESparsityError) as exc_info:
            diagonal_op_(bsr, offset=1)

        error_msg = str(exc_info.value)
        assert "offset" in error_msg.lower()


class TestSolverErrors:
    """Test solver error handling."""

    def test_cudss_unavailable_error(self):
        """Test CuDSS raises ImportError when unavailable."""
        # This test documents the expected behavior
        try:
            from bae.utils.pysolvers import CuDSS
            solver = CuDSS()
            # If we get here, CuDSS is available - that's fine
        except ImportError as e:
            # Expected when CuDSS is not installed
            assert "CuDSS" in str(e) or "cudss" in str(e).lower()


class TestDeviceConsistency:
    """Test device-related error handling."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_solver_device_mismatch_handling(self):
        """Test solvers handle device mismatches gracefully."""
        from bae.utils.pysolvers import PCG

        n = 10
        torch.manual_seed(42)

        # Create system on CPU
        M = torch.randn(n, n, dtype=torch.float64)
        A = (M.T @ M + torch.eye(n, dtype=torch.float64)).to_sparse_csr()
        b = torch.randn(n, dtype=torch.float64)

        solver = PCG(tol=1e-6, maxiter=100)

        # CPU solve should work
        x = solver(A, b)
        assert x.device.type == "cpu"

        # GPU solve should work
        A_gpu = A.to("cuda")
        b_gpu = b.to("cuda")
        x_gpu = solver(A_gpu, b_gpu)
        assert x_gpu.device.type == "cuda"
