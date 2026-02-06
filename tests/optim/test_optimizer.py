"""Tests for bae.optim.optimizer module."""
import pytest
import torch
import pypose as pp

from bae.autograd.function import TrackingTensor as Track
from bae.optim.optimizer import LM
from bae.utils.pysolvers import PCG


class TestLMOptimizer:
    """Tests for Levenberg-Marquardt optimizer."""

    def test_lm_verbose_flag_default(self):
        """Test that verbose flag defaults to False."""
        # Create a minimal mock model
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.Parameter(torch.zeros(2))

            def forward(self, input):
                return (self.x,)

            def loss(self, input, target):
                return self.x.sum()

        model = DummyModel()
        solver = PCG(tol=1e-6, maxiter=50)
        strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5)

        optimizer = LM(model, strategy=strategy, solver=solver)
        assert optimizer.verbose is False

    def test_lm_verbose_flag_true(self):
        """Test that verbose flag can be set to True."""
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.Parameter(torch.zeros(2))

            def forward(self, input):
                return (self.x,)

            def loss(self, input, target):
                return self.x.sum()

        model = DummyModel()
        solver = PCG(tol=1e-6, maxiter=50)
        strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5)

        optimizer = LM(model, strategy=strategy, solver=solver, verbose=True)
        assert optimizer.verbose is True

    def test_lm_has_mm_attribute(self):
        """Test that LM optimizer has mm (matrix multiply) attribute."""
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.Parameter(torch.zeros(2))

            def forward(self, input):
                return (self.x,)

            def loss(self, input, target):
                return self.x.sum()

        model = DummyModel()
        solver = PCG(tol=1e-6, maxiter=50)
        strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5)

        optimizer = LM(model, strategy=strategy, solver=solver)
        assert hasattr(optimizer, "mm")
        assert optimizer.mm is not None

    def test_lm_inherits_from_pypose(self):
        """Test that LM inherits from PyPose LevenbergMarquardt."""
        from pypose.optim import LevenbergMarquardt as ppLM

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.Parameter(torch.zeros(2))

            def forward(self, input):
                return (self.x,)

            def loss(self, input, target):
                return self.x.sum()

        model = DummyModel()
        solver = PCG(tol=1e-6, maxiter=50)
        strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5)

        optimizer = LM(model, strategy=strategy, solver=solver)
        assert isinstance(optimizer, ppLM)


class TestLMOptimizerWithTracking:
    """Test LM with TrackingTensor parameters."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_tracking_tensor_parameter(self, device):
        """Test that LM works with TrackingTensor parameters."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        class TrackedModel(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                data = torch.randn(5, 3, device=device, dtype=torch.float64)
                self.params = torch.nn.Parameter(Track(data))

            def forward(self, input):
                return (self.params,)

            def loss(self, input, target):
                return (self.params**2).sum() / 2

        model = TrackedModel(device).to(device)
        solver = PCG(tol=1e-6, maxiter=50)
        strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5)

        optimizer = LM(model, strategy=strategy, solver=solver, verbose=False)

        # Verify model parameter is TrackingTensor
        assert isinstance(model.params.data, Track)
