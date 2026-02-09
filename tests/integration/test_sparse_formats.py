"""Integration tests for sparse matrix format operations.

Tests BSR/BSC conversions and the optimized BSR×BSC multiplication kernel.
"""

import pytest
import torch

from bae.sparse.py_ops import bsr2bsc, sparse_bsr_bsc_mm
from bae.exceptions import BAESparsityError


def _create_bsr_matrix(m, n, block_size, density=0.3, dtype=torch.float64, device="cpu"):
    """Create a random BSR matrix for testing."""
    torch.manual_seed(42)

    mb, nb = m // block_size, n // block_size
    nnz_blocks = max(1, int(mb * nb * density))

    # Generate random block positions
    positions = torch.randperm(mb * nb)[:nnz_blocks]
    rows = positions // nb
    cols = positions % nb

    # Sort by row for CSR format
    sorted_indices = torch.argsort(rows * nb + cols)
    rows = rows[sorted_indices]
    cols = cols[sorted_indices]

    # Build crow_indices
    crow_indices = torch.zeros(mb + 1, dtype=torch.int32)
    for r in rows:
        crow_indices[r + 1:] += 1

    # Generate random block values
    values = torch.randn(nnz_blocks, block_size, block_size, dtype=dtype, device=device)

    return torch.sparse_bsr_tensor(
        crow_indices.to(device),
        cols.to(device).to(torch.int32),
        values,
        size=(m, n)
    )


class TestBsr2Bsc:
    """Test BSR to BSC conversion."""

    def test_basic_conversion(self):
        """Test basic BSR to BSC conversion preserves data."""
        bsr = _create_bsr_matrix(8, 8, 2, density=0.5)
        bsc = bsr2bsc(bsr)

        assert bsc.layout == torch.sparse_bsc
        assert bsc.shape == bsr.shape

        # Convert both to dense and compare
        bsr_dense = bsr.to_dense()
        bsc_dense = bsc.to_dense()
        assert torch.allclose(bsr_dense, bsc_dense)

    def test_rectangular_matrix(self):
        """Test BSR to BSC with rectangular matrix."""
        bsr = _create_bsr_matrix(8, 12, 2, density=0.5)
        bsc = bsr2bsc(bsr)

        assert bsc.layout == torch.sparse_bsc
        assert bsc.shape == bsr.shape

        bsr_dense = bsr.to_dense()
        bsc_dense = bsc.to_dense()
        assert torch.allclose(bsr_dense, bsc_dense)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_conversion_on_cuda(self):
        """Test BSR to BSC conversion on CUDA."""
        bsr = _create_bsr_matrix(8, 8, 2, density=0.5, device="cuda")
        bsc = bsr2bsc(bsr)

        assert bsc.device.type == "cuda"
        assert bsc.layout == torch.sparse_bsc

        bsr_dense = bsr.to_dense()
        bsc_dense = bsc.to_dense()
        assert torch.allclose(bsr_dense, bsc_dense)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for BSR×BSC kernel")
class TestSparseBsrBscMm:
    """Test BSR×BSC sparse matrix multiplication kernel."""

    def test_basic_multiplication(self):
        """Test basic BSR×BSC multiplication."""
        m, n, p = 8, 8, 8
        block_size = 2

        A_bsr = _create_bsr_matrix(m, n, block_size, density=0.5, device="cuda")
        B_bsr = _create_bsr_matrix(n, p, block_size, density=0.5, device="cuda")
        B_bsc = bsr2bsc(B_bsr)

        # Compute using kernel
        C = sparse_bsr_bsc_mm(A_bsr, B_bsc)

        # Compare with dense multiplication
        A_dense = A_bsr.to_dense()
        B_dense = B_bsr.to_dense()
        C_expected = A_dense @ B_dense
        C_actual = C.to_dense()

        assert torch.allclose(C_actual, C_expected, rtol=1e-4, atol=1e-6)

    def test_jtj_computation(self):
        """Test J^T @ J computation pattern used in optimization."""
        m, n = 16, 8  # Residuals x Parameters
        block_size = 2

        J_bsr = _create_bsr_matrix(m, n, block_size, density=0.5, device="cuda")

        # Compute J^T @ J using BSR×BSC path
        J_t_bsr = _create_bsr_matrix(n, m, block_size, density=0.5, device="cuda")
        J_bsc = bsr2bsc(J_bsr)

        # Note: This tests the kernel, not the full J^T @ J pattern
        # The actual J^T @ J would use J.mT which has different structure
        C = sparse_bsr_bsc_mm(J_t_bsr, J_bsc)

        # Verify output format
        assert C.layout == torch.sparse_bsr
        assert C.shape == (n, n)

    def test_format_validation(self):
        """Test that incorrect formats raise errors."""
        m, n = 8, 8
        block_size = 2

        A_bsr = _create_bsr_matrix(m, n, block_size, density=0.5, device="cuda")
        B_bsr = _create_bsr_matrix(n, n, block_size, density=0.5, device="cuda")

        # Should fail: second argument is BSR, not BSC
        with pytest.raises(BAESparsityError) as exc_info:
            sparse_bsr_bsc_mm(A_bsr, B_bsr)
        assert "BSC" in str(exc_info.value)

    def test_device_validation(self):
        """Test that CPU tensors raise errors."""
        m, n = 8, 8
        block_size = 2

        A_bsr = _create_bsr_matrix(m, n, block_size, density=0.5, device="cpu")
        B_bsr = _create_bsr_matrix(n, n, block_size, density=0.5, device="cpu")
        B_bsc = bsr2bsc(B_bsr)

        # Should fail: tensors are on CPU
        with pytest.raises(BAESparsityError) as exc_info:
            sparse_bsr_bsc_mm(A_bsr, B_bsc)
        assert "CUDA" in str(exc_info.value)


class TestSparseFormatPerformance:
    """Performance-related tests for sparse formats."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_bsr2bsc_not_slower_than_coo(self):
        """Test that BSR→BSC conversion is not significantly slower than COO path."""
        import time

        m, n = 1000, 1000
        block_size = 4

        J_bsr = _create_bsr_matrix(m, n, block_size, density=0.1, device="cuda")

        # Warm up
        for _ in range(3):
            _ = bsr2bsc(J_bsr)
            _ = J_bsr.to_sparse_coo()

        torch.cuda.synchronize()

        # Time BSR→BSC
        start = time.perf_counter()
        for _ in range(10):
            _ = bsr2bsc(J_bsr)
        torch.cuda.synchronize()
        bsc_time = time.perf_counter() - start

        # Time BSR→COO
        start = time.perf_counter()
        for _ in range(10):
            _ = J_bsr.to_sparse_coo()
        torch.cuda.synchronize()
        coo_time = time.perf_counter() - start

        # BSC conversion should not be more than 10x slower than COO
        # (generous margin since sub-ms timings are noisy)
        assert bsc_time < coo_time * 10, f"BSC: {bsc_time:.3f}s, COO: {coo_time:.3f}s"
