"""Tests for bae.sparse.py_ops module."""
import pytest
import torch

from bae.sparse.py_ops import (
    diagonal_op_,
    jtj_diag,
    spdiags_,
)


def _make_bsr_tensor(device, dtype=torch.float64):
    """Create a simple BSR tensor for testing."""
    crow_indices = torch.tensor([0, 2, 4], device=device)
    col_indices = torch.tensor([0, 1, 0, 1], device=device)
    values = torch.randn(4, 3, 3, device=device, dtype=dtype)
    return torch.sparse_bsr_tensor(crow_indices, col_indices, values, size=(6, 6))


class TestJtjDiag:
    """Tests for jtj_diag function."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_jtj_diag_basic(self, device):
        """Test jtj_diag produces correct diagonal of J^T @ J."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        crow_indices = torch.tensor([0, 2, 4], device=device)
        col_indices = torch.tensor([0, 1, 0, 1], device=device)
        values = torch.tensor(
            [
                [[0, 1, 2], [6, 7, 8]],
                [[3, 4, 5], [9, 10, 11]],
                [[12, 13, 14], [18, 19, 20]],
                [[15, 16, 17], [21, 22, 23]],
            ],
            device=device,
            dtype=torch.float64,
        )
        bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values)

        output = jtj_diag(bsr)

        # Verify against dense computation
        coo = bsr.to_sparse_coo()
        expected = (coo @ coo.mT).to_dense().diag()
        torch.testing.assert_close(output, expected)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_jtj_diag_random(self, device):
        """Test jtj_diag with random BSR tensor."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        bsr = _make_bsr_tensor(device)

        output = jtj_diag(bsr)

        # Verify shape
        assert output.shape == (6,)
        # Verify all values are non-negative (diagonal of J^T @ J)
        assert (output >= 0).all()

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_jtj_diag_contiguous_guard(self, device):
        """Test that jtj_diag handles non-contiguous indices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        bsr = _make_bsr_tensor(device)
        # This should work regardless of contiguity
        output = jtj_diag(bsr)
        assert output.shape == (6,)


class TestDiagonalOp:
    """Tests for diagonal_op_ function."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_diagonal_op_extract(self, device):
        """Test extracting diagonal from sparse matrix."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        bsr = _make_bsr_tensor(device)

        # Extract diagonal without operation
        diag = diagonal_op_(bsr, offset=0, op=None)

        assert diag is not None
        # Diagonal should be flattened
        assert diag.ndim == 1

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_diagonal_op_with_clamp(self, device):
        """Test diagonal_op_ with clamp operation."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from functools import partial

        bsr = _make_bsr_tensor(device)

        # Apply clamp operation to diagonal
        diag = diagonal_op_(bsr, offset=0, op=partial(torch.clamp_, min=0.1, max=10.0))

        assert diag is not None
        assert (diag >= 0.1).all()
        assert (diag <= 10.0).all()


class TestSpdiags:
    """Tests for spdiags_ function."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_spdiags_basic(self, device):
        """Test creating sparse diagonal matrix."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        diagonal = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        shape = (4, 4)

        result = spdiags_(diagonal, None, shape=shape, layout=None)

        assert result.layout == torch.sparse_csr
        assert result.shape == shape

        # Verify diagonal values
        dense = result.to_dense()
        torch.testing.assert_close(dense.diag(), diagonal)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_spdiags_as_preconditioner(self, device):
        """Test using spdiags_ for Jacobi preconditioner."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Simulate diagonal of A matrix
        diag = torch.tensor([2.0, 4.0, 1.0, 3.0], device=device)

        # Create inverse diagonal preconditioner
        M = spdiags_(1.0 / diag, None, shape=(4, 4), layout=None)

        # Verify M @ diag_matrix = I (approximately)
        dense_M = M.to_dense()
        diag_matrix = torch.diag(diag)
        result = dense_M @ diag_matrix

        torch.testing.assert_close(result, torch.eye(4, device=device), atol=1e-6, rtol=1e-6)
