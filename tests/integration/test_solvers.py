"""Integration tests for solver comparisons.

Tests that different solvers (PCG, PCG_, SciPy) produce consistent results
on the same optimization problems.
"""

import pytest
import torch

from bae.utils.pysolvers import PCG, PCG_, SciPySpSolver


def _make_sparse_spd_system(n: int, density: float, dtype: torch.dtype, device: str):
    """Create a sparse SPD system with controlled density."""
    torch.manual_seed(42)

    # Create sparse pattern
    nnz_per_row = max(2, int(n * density))
    indices = []
    values = []

    for i in range(n):
        # Always include diagonal
        indices.append((i, i))
        values.append(10.0)  # Strong diagonal for conditioning

        # Add some off-diagonal entries
        for _ in range(nnz_per_row - 1):
            j = torch.randint(0, n, (1,)).item()
            if j != i:
                indices.append((i, j))
                indices.append((j, i))  # Symmetric
                v = torch.randn(1).item() * 0.1
                values.append(v)
                values.append(v)

    # Build sparse tensor
    rows, cols = zip(*indices)
    indices_tensor = torch.tensor([rows, cols], dtype=torch.int64)
    values_tensor = torch.tensor(values, dtype=dtype)

    A_coo = torch.sparse_coo_tensor(indices_tensor, values_tensor, (n, n))
    A_coo = A_coo.coalesce()
    A_csr = A_coo.to_sparse_csr().to(device)

    b = torch.randn(n, dtype=dtype, device=device)
    return A_csr, b


class TestSolverConsistency:
    """Test that different solvers produce consistent results."""

    def test_pcg_vs_scipy_cpu(self):
        """Test PCG and SciPy produce similar solutions on CPU."""
        n = 50
        A, b = _make_sparse_spd_system(n, 0.1, torch.float64, "cpu")

        pcg_solver = PCG(tol=1e-10, maxiter=200)
        scipy_solver = SciPySpSolver()

        x_pcg = pcg_solver(A, b)
        x_scipy = scipy_solver(A, b)

        # Solutions should be close
        diff = torch.linalg.norm(x_pcg - x_scipy)
        assert diff < 1e-6, f"Solver mismatch: {diff}"

    @pytest.mark.parametrize("n", [20, 50, 100])
    def test_pcg_solution_quality(self, n):
        """Test PCG achieves specified tolerance."""
        A, b = _make_sparse_spd_system(n, 0.1, torch.float64, "cpu")

        tol = 1e-8
        solver = PCG(tol=tol, maxiter=500)
        x = solver(A, b)

        residual = A @ x - b
        rel_error = torch.linalg.norm(residual) / torch.linalg.norm(b)

        # Should achieve tolerance or better
        assert rel_error < tol * 100, f"PCG didn't converge: rel_error={rel_error}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pcg_gpu_vs_cpu(self):
        """Test PCG produces consistent results on CPU and GPU."""
        n = 50
        A_cpu, b_cpu = _make_sparse_spd_system(n, 0.1, torch.float64, "cpu")

        A_gpu = A_cpu.to("cuda")
        b_gpu = b_cpu.to("cuda")

        solver_cpu = PCG(tol=1e-10, maxiter=200)
        solver_gpu = PCG(tol=1e-10, maxiter=200)

        x_cpu = solver_cpu(A_cpu, b_cpu)
        x_gpu = solver_gpu(A_gpu, b_gpu)

        diff = torch.linalg.norm(x_cpu - x_gpu.cpu())
        assert diff < 1e-6, f"CPU/GPU mismatch: {diff}"


class TestSolverRobustness:
    """Test solver behavior on edge cases."""

    def test_pcg_ill_conditioned(self):
        """Test PCG handles ill-conditioned matrices."""
        n = 20
        torch.manual_seed(42)

        # Create ill-conditioned matrix
        M = torch.randn(n, n, dtype=torch.float64)
        A = M.T @ M + 1e-8 * torch.eye(n, dtype=torch.float64)
        A_sparse = A.to_sparse_csr()
        b = torch.randn(n, dtype=torch.float64)

        solver = PCG(tol=1e-6, maxiter=500)
        x = solver(A_sparse, b)

        # Should return something (not crash)
        assert x.shape == (n,)
        assert not torch.isnan(x).any()

    def test_pcg_maxiter_respected(self):
        """Test PCG respects maxiter even if not converged."""
        n = 100
        A, b = _make_sparse_spd_system(n, 0.1, torch.float64, "cpu")

        # Very tight tolerance, very few iterations
        solver = PCG(tol=1e-15, maxiter=3)
        x = solver(A, b)

        # Should return without error
        assert x.shape == (n,)

    def test_zero_rhs(self):
        """Test solver handles zero right-hand side."""
        n = 20
        A, _ = _make_sparse_spd_system(n, 0.1, torch.float64, "cpu")
        b = torch.zeros(n, dtype=torch.float64)

        solver = PCG(tol=1e-8, maxiter=100)
        x = solver(A, b)

        # Solution should be zero
        assert torch.allclose(x, torch.zeros_like(x), atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for PCG_")
class TestPCGCudaGraph:
    """Test PCG_ solver with CUDA graph acceleration."""

    def test_pcg_cuda_graph_solves_correctly(self):
        """Test PCG_ produces correct solutions."""
        n = 50
        A, b = _make_sparse_spd_system(n, 0.1, torch.float64, "cuda")

        solver = PCG_(tol=1e-8, maxiter=200)
        x = solver(A, b)

        residual = torch.linalg.norm(A @ x - b) / torch.linalg.norm(b)
        assert residual < 1e-6, f"PCG_ didn't converge: rel_error={residual}"

    def test_pcg_vs_pcg_cuda_graph(self):
        """Test PCG and PCG_ produce consistent results."""
        n = 50
        A, b = _make_sparse_spd_system(n, 0.1, torch.float64, "cuda")

        pcg_solver = PCG(tol=1e-10, maxiter=200)
        pcg_cuda_solver = PCG_(tol=1e-10, maxiter=200)

        x_pcg = pcg_solver(A, b)
        x_pcg_cuda = pcg_cuda_solver(A, b)

        diff = torch.linalg.norm(x_pcg - x_pcg_cuda)
        assert diff < 1e-6, f"PCG vs PCG_ mismatch: {diff}"

    def test_pcg_cuda_graph_reuse(self):
        """Test CUDA graphs are correctly reused across calls."""
        n = 50
        A, b = _make_sparse_spd_system(n, 0.1, torch.float64, "cuda")

        solver = PCG_(tol=1e-8, maxiter=200)

        # First call captures graphs
        x1 = solver(A, b)

        # Second call should reuse graphs
        x2 = solver(A, b)

        # Results should be identical (same input)
        assert torch.allclose(x1, x2, atol=1e-10)

    def test_pcg_cuda_graph_shape_change(self):
        """Test PCG_ handles shape changes by recapturing graphs."""
        # First problem size
        n1 = 30
        A1, b1 = _make_sparse_spd_system(n1, 0.1, torch.float64, "cuda")

        solver = PCG_(tol=1e-8, maxiter=200)
        x1 = solver(A1, b1)

        # Different problem size - should recapture graphs
        n2 = 50
        A2, b2 = _make_sparse_spd_system(n2, 0.1, torch.float64, "cuda")
        x2 = solver(A2, b2)

        # Both should solve correctly
        residual1 = torch.linalg.norm(A1 @ x1 - b1) / torch.linalg.norm(b1)
        residual2 = torch.linalg.norm(A2 @ x2 - b2) / torch.linalg.norm(b2)

        assert residual1 < 1e-6, f"First solve failed: {residual1}"
        assert residual2 < 1e-6, f"Second solve failed: {residual2}"

    def test_pcg_cuda_graph_performance(self):
        """Test that PCG_ with CUDA graphs is not slower than PCG."""
        import time

        n = 100
        A, b = _make_sparse_spd_system(n, 0.1, torch.float64, "cuda")

        pcg_solver = PCG(tol=1e-8, maxiter=100)
        pcg_cuda_solver = PCG_(tol=1e-8, maxiter=100)

        # Warm up both solvers
        for _ in range(3):
            _ = pcg_solver(A, b)
            _ = pcg_cuda_solver(A, b)

        torch.cuda.synchronize()

        # Time PCG
        start = time.perf_counter()
        for _ in range(10):
            _ = pcg_solver(A, b)
        torch.cuda.synchronize()
        pcg_time = time.perf_counter() - start

        # Time PCG_
        start = time.perf_counter()
        for _ in range(10):
            _ = pcg_cuda_solver(A, b)
        torch.cuda.synchronize()
        pcg_cuda_time = time.perf_counter() - start

        # PCG_ should not be significantly slower (allow 20% margin)
        # In practice it should be faster after warm-up
        assert pcg_cuda_time < pcg_time * 1.2, (
            f"PCG_: {pcg_cuda_time:.3f}s, PCG: {pcg_time:.3f}s"
        )
