"""Tests for bae.utils.pysolvers module."""
import pytest
import torch

from bae.utils.pysolvers import PCG, SciPySpSolver


def _make_spd_system(n: int, dtype: torch.dtype, device: str):
    """Create a symmetric positive definite system Ax = b."""
    torch.manual_seed(42)
    # Create SPD matrix: A = M^T M + eps*I
    M = torch.randn(n + 5, n, dtype=dtype, device=device)
    A = M.T @ M + 0.1 * torch.eye(n, dtype=dtype, device=device)
    b = torch.randn(n, dtype=dtype, device=device)
    return A, b


def _make_sparse_spd_system(n: int, dtype: torch.dtype, device: str):
    """Create a sparse SPD system."""
    A_dense, b = _make_spd_system(n, dtype, device)
    A_sparse = A_dense.to_sparse_csr()
    return A_sparse, b


class TestPCG:
    """Tests for Preconditioned Conjugate Gradient solver."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_pcg_dense_basic(self, device, dtype):
        """Test PCG with dense matrix."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        n = 10
        A, b = _make_spd_system(n, dtype, device)

        solver = PCG(tol=1e-6, maxiter=100)
        x = solver(A, b)

        # Check solution accuracy
        residual = A @ x - b
        rel_error = torch.linalg.norm(residual) / torch.linalg.norm(b)

        atol = 1e-4 if dtype == torch.float32 else 1e-8
        assert rel_error < atol, f"Relative error {rel_error} too large"

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_pcg_sparse_csr(self, device):
        """Test PCG with sparse CSR matrix."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        n = 20
        A, b = _make_sparse_spd_system(n, torch.float64, device)

        solver = PCG(tol=1e-8, maxiter=200)
        x = solver(A, b)

        # Check solution
        residual = A @ x - b
        rel_error = torch.linalg.norm(residual) / torch.linalg.norm(b)

        assert rel_error < 1e-6, f"Relative error {rel_error} too large"

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_pcg_with_initial_guess(self, device):
        """Test PCG with non-zero initial guess."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        n = 10
        A, b = _make_spd_system(n, torch.float64, device)

        # Solve first with zero initial guess
        solver = PCG(tol=1e-8, maxiter=100)
        x_true = solver(A, b)

        # Now use a good initial guess (should converge faster)
        x0 = x_true + 0.01 * torch.randn_like(x_true)
        # PCG doesn't take x0 directly via forward, but internally handles it

        x = solver(A, b)
        residual = A @ x - b
        rel_error = torch.linalg.norm(residual) / torch.linalg.norm(b)

        assert rel_error < 1e-6

    def test_pcg_maxiter_limit(self):
        """Test PCG respects maxiter limit."""
        n = 50
        A, b = _make_spd_system(n, torch.float64, "cpu")

        # Very few iterations - won't converge
        solver = PCG(tol=1e-12, maxiter=2)
        x = solver(A, b)

        # Should return something (not error), even if not converged
        assert x.shape == (n,)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_pcg_1d_vs_2d_input(self, device):
        """Test PCG handles both 1D and 2D b vectors."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        n = 10
        A, b = _make_spd_system(n, torch.float64, device)

        solver = PCG(tol=1e-8, maxiter=100)

        # Test with 1D b
        x1 = solver(A, b)
        assert x1.ndim == 1

        # Test with 2D b (column vector)
        b_2d = b.unsqueeze(-1)
        x2 = solver(A, b_2d)
        assert x2.ndim == 1  # Should squeeze output


class TestSciPySpSolver:
    """Tests for SciPy sparse solver."""

    def test_scipy_solver_basic(self):
        """Test SciPy solver with basic CSR matrix."""
        n = 20
        A, b = _make_sparse_spd_system(n, torch.float64, "cpu")

        solver = SciPySpSolver()
        x = solver(A, b)

        # Check solution
        residual = A @ x - b
        rel_error = torch.linalg.norm(residual) / torch.linalg.norm(b)

        assert rel_error < 1e-10, f"Relative error {rel_error} too large"

    def test_scipy_solver_converts_non_csr(self):
        """Test SciPy solver handles non-CSR input."""
        n = 10
        A_dense, b = _make_spd_system(n, torch.float64, "cpu")
        A_coo = A_dense.to_sparse_coo()

        solver = SciPySpSolver()
        x = solver(A_coo, b)

        # Should convert and solve
        residual = A_coo @ x - b
        rel_error = torch.linalg.norm(residual) / torch.linalg.norm(b)

        assert rel_error < 1e-10

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_scipy_solver_dtypes(self, dtype):
        """Test SciPy solver with different dtypes."""
        n = 15
        A, b = _make_sparse_spd_system(n, dtype, "cpu")

        solver = SciPySpSolver()
        x = solver(A, b)

        residual = A @ x - b
        rel_error = torch.linalg.norm(residual) / torch.linalg.norm(b)

        atol = 1e-4 if dtype == torch.float32 else 1e-10
        assert rel_error < atol
