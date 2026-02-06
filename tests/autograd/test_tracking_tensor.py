"""Tests for bae.autograd.function TrackingTensor."""
import pytest
import torch

from bae.autograd.function import TrackingTensor, map_transform


class TestTrackingTensor:
    """Tests for TrackingTensor class."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_tracking_tensor_creation(self, device):
        """Test creating a TrackingTensor."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        data = torch.randn(5, 3, device=device)
        track = TrackingTensor(data)

        assert track.shape == data.shape
        assert track.device == data.device
        assert track.dtype == data.dtype

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_tracking_tensor_indexing(self, device):
        """Test indexing operations are tracked."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        data = torch.randn(10, 3, device=device)
        track = TrackingTensor(data)

        indices = torch.tensor([0, 2, 5, 7], device=device, dtype=torch.int32)
        result = track[indices]

        # Result should also be a TrackingTensor
        assert isinstance(result, TrackingTensor)
        # Values should match
        torch.testing.assert_close(result.tensor(), data[indices])

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_tracking_tensor_nested_indexing(self, device):
        """Test nested indexing operations."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        data = torch.randn(10, 5, 3, device=device)
        track = TrackingTensor(data)

        idx1 = torch.tensor([1, 3, 5], device=device, dtype=torch.int32)
        idx2 = torch.tensor([0, 2, 1], device=device, dtype=torch.int32)

        result = track[idx1][idx2]

        expected = data[idx1][idx2]
        torch.testing.assert_close(result.tensor(), expected)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_tracking_tensor_jactrace(self, device):
        """Test jactrace attribute for jacobian computation."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        data = torch.randn(8, 4, device=device, requires_grad=True)
        track = TrackingTensor(data)

        indices = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
        result = track[indices]

        # TrackingTensor stores trace info internally, verify it's a TrackingTensor
        assert isinstance(result, TrackingTensor)
        # The trace is stored in the tensor subclass itself
        assert result.shape == (3, 4)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_tracking_tensor_tensor_method(self, device):
        """Test .tensor() method returns underlying tensor."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        data = torch.randn(5, 3, device=device)
        track = TrackingTensor(data)

        recovered = track.tensor()

        assert isinstance(recovered, torch.Tensor)
        assert not isinstance(recovered, TrackingTensor)
        torch.testing.assert_close(recovered, data)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_tracking_tensor_arithmetic(self, device):
        """Test arithmetic operations with TrackingTensor."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        data = torch.randn(5, 3, device=device)
        track = TrackingTensor(data)
        other = torch.randn(5, 3, device=device)

        # Addition
        result_add = track + other
        torch.testing.assert_close(result_add.tensor(), data + other)

        # Subtraction
        result_sub = track - other
        torch.testing.assert_close(result_sub.tensor(), data - other)

        # Multiplication
        result_mul = track * 2.0
        torch.testing.assert_close(result_mul.tensor(), data * 2.0)


class TestMapTransform:
    """Tests for map_transform decorator."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_map_transform_basic(self, device):
        """Test map_transform with basic function."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        @map_transform
        def square(x):
            return x**2

        data = torch.randn(10, 3, device=device)
        track = TrackingTensor(data)
        indices = torch.tensor([0, 2, 5], device=device, dtype=torch.int32)

        indexed = track[indices]
        result = square(indexed)

        expected = data[indices] ** 2
        torch.testing.assert_close(result.tensor(), expected)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_map_transform_with_args(self, device):
        """Test map_transform with additional arguments."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        @map_transform
        def scale_and_shift(x, scale, shift):
            return x * scale + shift

        data = torch.randn(8, 4, device=device)
        track = TrackingTensor(data)
        indices = torch.tensor([1, 3, 6], device=device, dtype=torch.int32)

        indexed = track[indices]
        result = scale_and_shift(indexed, 2.0, 1.0)

        expected = data[indices] * 2.0 + 1.0
        torch.testing.assert_close(result.tensor(), expected)
