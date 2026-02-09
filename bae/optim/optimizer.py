"""Levenberg-Marquardt optimizer for sparse nonlinear least squares.

This module provides an optimized LM implementation that uses sparse block
matrices (BSR format) for efficient jacobian computation on GPU.
"""

from functools import partial
import logging
import math
import torch
from pypose.optim import LevenbergMarquardt as ppLM
import pypose as pp
from ..autograd.graph import jacobian
from ..autograd.function import TrackingTensor
from ..sparse.py_ops import (
    diagonal_op_, jtj_diag, bsr_hcat, bsr2bsc, bsc2bsr,
    sparse_bsr_bsc_mm, is_bsr_bsc_mm_available
)
from ..exceptions import BAESolverError

# CUDA extension required - no fallback
from ..sparse.spgemm import CuSparse

_logger = logging.getLogger(__name__)

# Control flag for BSR path (can be disabled for debugging)
# Will automatically fall back to CSR if BSR×BSC kernel is unavailable
USE_BSR_PATH = True
_BSR_PATH_AVAILABLE = None  # Lazy-checked on first use


# --- Auto-selection infrastructure ---

class _DeferredSolver:
    """Placeholder solver passed to ppLM.__init__ when solver is deferred.

    Replaced with a real solver at the first step() call.
    """
    def __call__(self, A, b, **kwargs):
        raise RuntimeError("_DeferredSolver should have been replaced before first solve")


class _DeferredStrategy:
    """Placeholder strategy passed to ppLM.__init__ when strategy is deferred.

    Has a no-op update() and default parameters matching TrustRegion so that
    ppLM.__init__ can initialize parameter groups correctly.
    Replaced with a real strategy at the first step() call.
    """
    defaults = {
        'radius': 1e6, 'damping': 1e-6, 'high': 0.5, 'low': 0.001,
        'up': 2.0, 'down': 0.5, 'factor': 0.5,
    }

    def update(self, *args, **kwargs):
        pass


def _resolve_solver(name, device):
    """Map a solver name string to a solver instance.

    Parameters
    ----------
    name : str
        One of "auto", "pcg", "pcg_", "cudss".
    device : torch.device or str
        Target device, used for "auto" selection.

    Returns
    -------
    object
        A solver instance.
    """
    from ..utils.pysolvers import PCG, PCG_, CuDSS

    if name == "auto":
        if str(device).startswith("cuda"):
            return PCG_(tol=1e-4, maxiter=250)
        else:
            return PCG(tol=1e-4, maxiter=250)
    elif name == "pcg":
        return PCG(tol=1e-4, maxiter=250)
    elif name == "pcg_":
        return PCG_(tol=1e-4, maxiter=250)
    elif name == "cudss":
        return CuDSS()
    else:
        raise ValueError(
            f"Unknown solver '{name}'. Choose from: 'auto', 'pcg', 'pcg_', 'cudss'"
        )


def _resolve_strategy(name):
    """Map a strategy name string to a strategy instance.

    Parameters
    ----------
    name : str
        One of "auto", "trustregion", "adaptive".

    Returns
    -------
    object
        A strategy instance.
    """
    from ..utils.schur import TrustRegion, Adaptive

    if name in ("auto", "trustregion"):
        return TrustRegion(up=2.0, down=0.5**4)
    elif name == "adaptive":
        return Adaptive()
    else:
        raise ValueError(
            f"Unknown strategy '{name}'. Choose from: 'auto', 'trustregion', 'adaptive'"
        )


def _prepare_jacobian_for_spgemm(jacobians):
    """Prepare jacobian tensors for sparse matrix multiplication.

    Converts BSR jacobians to CSR format required by cuSparse SpGEMM.

    Note: Current implementation uses COO as intermediate format because
    torch.cat for sparse tensors requires COO. Future optimization could
    implement direct BSR/CSR concatenation.

    Parameters
    ----------
    jacobians : list of torch.Tensor
        List of BSR sparse jacobian matrices from jacobian().

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor)
        (J, J_T) - Jacobian and its transpose in CSR format.
    """
    # BSR → COO (required for torch.cat)
    J_coo = torch.cat([j.to_sparse_coo() for j in jacobians], dim=-1)

    # COO → CSR (required by cuSparse SpGEMM)
    # Note: Computing transpose before CSR conversion avoids extra work
    J = J_coo.to_sparse_csr()
    J_T = J_coo.mT.to_sparse_csr()

    return J, J_T


def _prepare_jacobian_bsr(jacobians):
    """Prepare jacobian tensors using BSR path (no COO conversion).

    Horizontally concatenates BSR jacobians and prepares for BSR×BSC
    multiplication. This avoids the COO conversion overhead.

    Parameters
    ----------
    jacobians : list of torch.Tensor
        List of BSR sparse jacobian matrices from jacobian().

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
        (J_bsr, J_T_bsr, J_bsc) - Jacobian in BSR, its transpose in BSR,
        and the original in BSC format.
    """
    # Horizontal concatenation of BSR matrices (no COO conversion)
    J_bsr = bsr_hcat(jacobians)

    # Convert to BSC for efficient J^T @ J computation
    J_bsc = bsr2bsc(J_bsr)

    # Get J^T in BSR format: transpose of BSC is BSR
    # We use bsc2bsr for explicit conversion to ensure proper layout
    J_T_bsc = J_bsr.mT  # Transpose of BSR gives BSC view
    J_T_bsr = bsc2bsr(J_T_bsc)

    return J_bsr, J_T_bsr, J_bsc


def _compute_jtj_bsr(J_T_bsr, J_bsc):
    """Compute J^T @ J using BSR×BSC kernel.

    Parameters
    ----------
    J_T_bsr : torch.Tensor
        J^T in BSR format (params × residuals).
    J_bsc : torch.Tensor
        J in BSC format (residuals × params).

    Returns
    -------
    torch.Tensor
        J^T @ J matrix in BSR format.
    """
    return sparse_bsr_bsc_mm(J_T_bsr, J_bsc)


class LM(ppLM):
    """Levenberg-Marquardt optimizer with sparse jacobian support.

    Extends PyPose's LM optimizer with efficient sparse block matrix operations
    using custom CUDA kernels for J^T @ J computation.

    Supports auto-selection of method, solver, and strategy based on the
    problem structure detected at the first ``step()`` call.

    Parameters
    ----------
    model : torch.nn.Module
        The model to optimize. Must have a ``loss()`` method.
    solver : str or callable, optional
        Linear solver. Can be a solver object (e.g., ``PCG()``) or a string:
        ``"auto"`` (default), ``"pcg"``, ``"pcg_"``, ``"cudss"``.
        When ``"auto"``, selects ``PCG_`` on CUDA, ``PCG`` on CPU.
    strategy : str or object, optional
        Damping strategy. Can be a strategy object or a string:
        ``"auto"`` (default), ``"trustregion"``, ``"adaptive"``.
    method : str, optional
        Optimization method: ``"auto"`` (default), ``"standard"``, ``"schur"``.
        When ``"auto"``, detects BA structure and chooses accordingly.
    verbose : bool, optional
        If True, log optimization progress. Default is False.
    **kwargs
        Additional arguments passed to parent LevenbergMarquardt
        (e.g., ``reject``, ``min``, ``max``).

    Examples
    --------
    >>> # Full auto-selection (recommended)
    >>> optimizer = LM(model)
    >>> loss = optimizer.step(input_data)

    >>> # String-based configuration
    >>> optimizer = LM(model, solver="pcg", strategy="adaptive", method="schur")

    >>> # Object-based (backward-compatible)
    >>> from bae.utils.pysolvers import PCG
    >>> optimizer = LM(model, solver=PCG(maxiter=100))
    """

    def __init__(self, model, solver="auto", strategy="auto", method="auto",
                 verbose=False, **kwargs):
        # Store auto-selection config
        self._method = method
        self._solver_config = solver if isinstance(solver, str) else None
        self._strategy_config = strategy if isinstance(strategy, str) else None
        self._resolved = not (isinstance(solver, str) or isinstance(strategy, str))

        # Build the actual solver/strategy to pass to ppLM.__init__
        if isinstance(solver, str):
            init_solver = _DeferredSolver()
        else:
            init_solver = solver

        if isinstance(strategy, str):
            init_strategy = _DeferredStrategy()
        else:
            init_strategy = strategy

        super(LM, self).__init__(model, solver=init_solver, strategy=init_strategy, **kwargs)
        self.mm = CuSparse()
        self.verbose = verbose

    def _resolve_auto(self, J_bsr_list, pg):
        """Resolve deferred solver, strategy, and method at first step().

        Called once when ``self._resolved`` is False. Inspects the jacobian
        structure and device to choose optimal defaults.

        Parameters
        ----------
        J_bsr_list : list of torch.Tensor
            BSR jacobian matrices from the first forward pass.
        pg : dict
            Parameter group (used to inspect params for BA structure).
        """
        device = J_bsr_list[0].device if J_bsr_list else torch.device('cpu')

        # Resolve solver
        if self._solver_config is not None:
            self.solver = _resolve_solver(self._solver_config, device)
            _logger.info(f"Auto-selected solver: {type(self.solver).__name__}")

        # Resolve strategy
        if self._strategy_config is not None:
            self.strategy = _resolve_strategy(self._strategy_config)
            _logger.info(f"Auto-selected strategy: {type(self.strategy).__name__}")

        # Resolve method
        if self._method == "auto":
            has_ba_structure = (
                len(J_bsr_list) == 2
                and any(getattr(p, 'trim_SE3_grad', False) for p in pg['params'])
            )
            if has_ba_structure:
                # Estimate W memory: n_cam_params * 3 * n_points * 4 bytes
                n_cam_params = J_bsr_list[0].shape[1]
                n_points = J_bsr_list[1].shape[1] // 3
                W_mem_bytes = n_cam_params * 3 * n_points * 4
                if W_mem_bytes < 4 * 1024**3:  # < 4 GB
                    self._method = "schur"
                else:
                    self._method = "standard"
                _logger.info(
                    f"Auto-detected BA structure ({n_cam_params} cam params, "
                    f"{n_points} points, W≈{W_mem_bytes / 1024**2:.0f}MB). "
                    f"Selected method: {self._method}"
                )
            else:
                self._method = "standard"
                _logger.info("No BA structure detected. Selected method: standard")

        self._resolved = True

    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        """Perform one optimization step.

        On the first call, resolves any deferred solver/strategy/method
        choices based on the problem structure.

        Parameters
        ----------
        input : torch.Tensor
            Input data for the model.
        target : torch.Tensor, optional
            Target data for the model.
        weight : torch.Tensor, optional
            Per-residual weights.

        Returns
        -------
        float
            The loss value after this step.

        Raises
        ------
        BAESolverError
            If the linear solver fails after all retry attempts.
        """
        for pg in self.param_groups:
            weight = self.weight if weight is None else weight
            R = list(self.model(input, target))
            R = R[0]
            J_bsr_list = jacobian(R, pg['params'])
            if isinstance(R, TrackingTensor):
                R = R.tensor()

            # Resolve deferred auto-selection on first step
            if not self._resolved:
                self._resolve_auto(J_bsr_list, pg)

            self.last = self.loss = self.loss if hasattr(self, 'loss') else self.model.loss(input, target)
            self.reject_count = 0

            # Dispatch to the appropriate method
            if self._method == "schur":
                return self._step_schur(input, target, weight, J_bsr_list, R, pg)
            else:
                return self._step_standard(input, target, weight, J_bsr_list, R, pg)
        return self.loss

    def _step_standard(self, input, target, weight, J_bsr_list, R, pg):
        """Standard LM step using full J^T J system."""
        # Use BSR path if enabled, kernel available, and on CUDA
        global _BSR_PATH_AVAILABLE
        if _BSR_PATH_AVAILABLE is None:
            _BSR_PATH_AVAILABLE = is_bsr_bsc_mm_available()
            if USE_BSR_PATH and not _BSR_PATH_AVAILABLE:
                _logger.info(
                    "BSR×BSC kernel not available, using CSR path. "
                    "Rebuild bae with CUDA extensions for optimal performance."
                )

        # Check if BSR path can be used
        can_use_bsr = (
            USE_BSR_PATH
            and _BSR_PATH_AVAILABLE
            and J_bsr_list
            and J_bsr_list[0].is_cuda
            and all(j.layout == torch.sparse_bsr for j in J_bsr_list)
        )

        if can_use_bsr and len(J_bsr_list) > 1:
            first_block_shape = J_bsr_list[0].values().shape[-2:]
            can_use_bsr = all(
                j.values().shape[-2:] == first_block_shape
                for j in J_bsr_list
            )

        if can_use_bsr:
            J_bsr, J_T_bsr, J_bsc = _prepare_jacobian_bsr(J_bsr_list)
            A = _compute_jtj_bsr(J_T_bsr, J_bsc)
            # Keep BSR format: diagonal_op_, PCG, and matmul all support BSR
            J = J_bsr
            J_T = J_T_bsr
        else:
            J, J_T = _prepare_jacobian_for_spgemm(J_bsr_list)
            A = self.mm(J_T, J)

        diagonal_op_(A, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))

        while self.last <= self.loss:
            diagonal_op_(A, op=partial(torch.mul, other=1+pg['damping']))
            try:
                D = self.solver(A, -J_T @ R.view(-1, 1))
                D = D[:, None]
            except (RuntimeError, ValueError) as e:
                _logger.warning(f"Linear solver failed: {e}")
                raise BAESolverError(
                    f"Linear solver failed during optimization step. "
                    f"This may indicate a singular or ill-conditioned matrix. "
                    f"Try increasing damping or using a different solver. "
                    f"Original error: {e}"
                ) from e
            self.update_parameter(pg['params'], D)
            self.loss = self.model.loss(input, target)
            if self.verbose:
                _logger.info(
                    f"Loss: {self.loss:.6e}, Last: {self.last:.6e}, "
                    f"Rejects: {self.reject_count}, Damping: {pg['damping']:.6e}"
                )
            self.strategy.update(pg, last=self.last, loss=self.loss, J=J, D=D, R=R.view(-1, 1))
            if self.last < self.loss and self.reject_count < self.reject:
                self.update_parameter(params=pg['params'], step=-D)
                self.loss, self.reject_count = self.last, self.reject_count + 1
            else:
                break
        return self.loss

    def _step_schur(self, input, target, weight, J_bsr_list, R, pg):
        """LM step using Schur complement elimination."""
        if len(J_bsr_list) != 2:
            _logger.warning(
                f"Schur method expects 2 jacobians (cameras, points), got {len(J_bsr_list)}. "
                "Falling back to standard."
            )
            return self._step_standard(input, target, weight, J_bsr_list, R, pg)

        Jc_bsr, Jp_bsr = J_bsr_list

        Jc_coo = Jc_bsr.to_sparse_coo()
        Jp_coo = Jp_bsr.to_sparse_coo()
        Jc = Jc_coo.to_sparse_csr()
        Jp = Jp_coo.to_sparse_csr()

        # Full J for strategy update
        J = torch.cat([Jc_coo, Jp_coo], dim=-1).to_sparse_csr()

        while self.last <= self.loss:
            try:
                D = self._solve_schur(
                    Jc, Jp, R.view(-1),
                    damping=pg['damping'],
                    min_diag=pg['min'],
                    max_diag=pg['max']
                )
                D = D[:, None]
            except (RuntimeError, ValueError) as e:
                _logger.warning(f"Schur solver failed: {e}")
                raise BAESolverError(
                    f"Schur complement solver failed. "
                    f"Original error: {e}"
                ) from e

            self.update_parameter(pg['params'], D)
            self.loss = self.model.loss(input, target)

            if self.verbose:
                _logger.info(
                    f"Loss: {self.loss:.6e}, Last: {self.last:.6e}, "
                    f"Rejects: {self.reject_count}, Damping: {pg['damping']:.6e}"
                )

            self.strategy.update(pg, last=self.last, loss=self.loss, J=J, D=D, R=R.view(-1, 1))

            if self.last < self.loss and self.reject_count < self.reject:
                self.update_parameter(params=pg['params'], step=-D)
                self.loss, self.reject_count = self.last, self.reject_count + 1
            else:
                break

        return self.loss

    def update_parameter(self, params, step):
        numels = []
        for param in params:
            if param.requires_grad:
                if getattr(param, 'trim_SE3_grad', False):
                    numels.append(math.prod(param.shape[:-1]) * (param.shape[-1] - 1))
                else:
                    numels.append(param.numel())
        steps = step.split(numels)
        for (param, d) in zip(params, steps):
            if param.requires_grad:
                if getattr(param, 'trim_SE3_grad', False):
                    param[..., :7] = pp.SE3(param[..., :7]).add_(pp.se3(d.view(param.shape[0], -1)[..., :6]))
                    if param.shape[-1] > 7:
                        param[:, 7:] += d.view(param.shape[0], -1)[:, 6:]
                else:
                    param.add_(d.view(param.shape))

    # --- Schur complement methods (used when method="schur") ---

    def _solve_schur(self, Jc, Jp, r, damping, min_diag, max_diag):
        """Solve normal equations using Schur complement elimination.

        Uses implicit Schur complement: never forms S explicitly.
        Instead, defines S @ x = U @ x - W @ (V^{-1} @ (W^T @ x))
        and runs PCG using this matvec operator.

        Parameters
        ----------
        Jc : torch.Tensor
            Camera jacobian in CSR format.
        Jp : torch.Tensor
            Point jacobian in CSR format.
        r : torch.Tensor
            Residual vector.
        damping : float
            LM damping factor.
        min_diag, max_diag : float
            Diagonal clamping bounds.

        Returns
        -------
        torch.Tensor
            Combined update vector [Δc; Δp].
        """
        # Compute transposes
        Jc_coo = Jc.to_sparse_coo()
        Jp_coo = Jp.to_sparse_coo()
        Jc_T = Jc_coo.mT.to_sparse_csr()
        Jp_T = Jp_coo.mT.to_sparse_csr()

        n_cam_params = Jc.shape[1]
        n_points = Jp.shape[1] // 3

        # Compute V^{-1} blocks (V is block-diagonal, trivial to invert)
        V_inv = self._compute_V_inverse(Jp, n_points, damping, min_diag, max_diag)

        def apply_V_inv(v):
            """Apply V^{-1} to vector v of shape (3*n_points, 1)."""
            return torch.bmm(V_inv, v.view(n_points, 3, 1)).view(-1, 1)

        # RHS vectors
        bc = -(Jc_T @ r.view(-1, 1))
        bp = -(Jp_T @ r.view(-1, 1))

        # U = Jc^T @ Jc (with damping)
        U = self.mm(Jc_T, Jc)
        diagonal_op_(U, op=partial(torch.clamp_, min=min_diag, max=max_diag))
        diagonal_op_(U, op=partial(torch.mul, other=1+damping))

        # Decide: explicit S + direct solve vs implicit PCG
        # based on W's dense memory footprint
        W_mem_bytes = n_cam_params * 3 * n_points * 4
        use_explicit = W_mem_bytes < 4 * 1024**3  # < 4 GB

        if use_explicit:
            # Small/medium: form S explicitly, direct solve
            W = torch.sparse.mm(Jc_T, Jp)  # CSR result
            V_inv_bp = apply_V_inv(bp)
            rhs = bc - torch.sparse.mm(W, V_inv_bp)

            # Compute S = U - W @ V^{-1} @ W^T using sparse-dense matmul
            # V_inv_W_T: apply V^{-1} to W^T blocks, need W^T dense
            W_T_dense = W.to_dense().T  # (3*n_points, n_cam_params)
            W_T_blocks = W_T_dense.reshape(n_points, 3, n_cam_params)
            V_inv_W_T = torch.bmm(V_inv, W_T_blocks).reshape(n_points * 3, n_cam_params)

            # W @ V_inv_W_T using sparse W (much faster than dense @ dense)
            S = U.to_dense() - torch.sparse.mm(W, V_inv_W_T)

            delta_c = torch.linalg.solve(S, rhs)

            # Back-substitute using W^T (dense)
            WT_dc = W_T_dense @ delta_c
            delta_p = apply_V_inv(bp - WT_dc).view(-1)
        else:
            # Large: implicit Schur complement + PCG
            def W_matvec(x):
                return Jc_T @ (Jp @ x)
            def WT_matvec(x):
                return Jp_T @ (Jc @ x)

            V_inv_bp = apply_V_inv(bp)
            rhs = bc - W_matvec(V_inv_bp)

            def schur_matvec(x):
                return (U @ x) - W_matvec(apply_V_inv(WT_matvec(x)))

            U_diag = U.diagonal().clone()
            U_diag[U_diag.abs() < 1e-6] = 1e-6
            M_diag = 1.0 / U_diag

            delta_c = self._pcg_implicit(schur_matvec, rhs, M_diag)

            delta_p = apply_V_inv(bp - WT_matvec(delta_c.view(-1, 1))).view(-1)

        return torch.cat([delta_c.view(-1), delta_p])

    def _pcg_implicit(self, matvec, b, M_diag, tol=1e-4, maxiter=250):
        """PCG solver with implicit matrix-vector product.

        Parameters
        ----------
        matvec : callable
            Function computing A @ x for the implicit matrix.
        b : torch.Tensor
            RHS vector (n, 1).
        M_diag : torch.Tensor
            Diagonal preconditioner values (n,).
        tol : float
            Convergence tolerance.
        maxiter : int
            Maximum iterations.

        Returns
        -------
        torch.Tensor
            Solution vector (n, 1).
        """
        x = torch.zeros_like(b)
        r = b.clone()
        bnrm2 = torch.linalg.norm(b)
        if bnrm2 == 0:
            return x
        atol = tol * bnrm2

        # Apply preconditioner: z = M @ r
        z = M_diag.view(-1, 1) * r
        p = z.clone()
        rho = (r * z).sum()

        for _ in range(maxiter):
            if torch.linalg.norm(r) < atol:
                break

            q = matvec(p)
            alpha = rho / (p * q).sum()
            x = x + alpha * p
            r = r - alpha * q

            z = M_diag.view(-1, 1) * r
            rho_new = (r * z).sum()
            beta = rho_new / rho
            p = z + beta * p
            rho = rho_new

        return x

    def _compute_V_inverse(self, Jp, n_points, damping, min_diag, max_diag):
        """Compute V^{-1} where V = Jp^T @ Jp + damping*I.

        V is block-diagonal (3×3 blocks) because each point only affects
        observations of that point. Fully vectorized using scatter_add.
        """
        device = Jp.device
        dtype = Jp.dtype

        crow = Jp.crow_indices()
        col = Jp.col_indices()
        val = Jp.values()

        # In BA, each residual row of Jp has exactly 3 non-zeros
        # (derivatives w.r.t. one point's x, y, z)
        row_starts = crow[:-1]
        row_ends = crow[1:]
        row_nnz = row_ends - row_starts
        non_empty = row_nnz > 0
        ne_starts = row_starts[non_empty]

        # Extract 3 values per non-empty row
        row_vals = torch.stack([val[ne_starts], val[ne_starts + 1], val[ne_starts + 2]], dim=1)

        # Point index per row
        point_idx = col[ne_starts] // 3

        # Compute outer products: (n_rows, 3, 3)
        outer = row_vals.unsqueeze(2) * row_vals.unsqueeze(1)

        # Accumulate into V blocks using scatter_add
        V_blocks = torch.zeros(n_points, 3, 3, device=device, dtype=dtype)
        idx = point_idx.view(-1, 1, 1).expand_as(outer)
        V_blocks.scatter_add_(0, idx, outer)

        # Apply damping and clamping
        diag_vals = V_blocks.diagonal(dim1=-2, dim2=-1)
        diag_vals.clamp_(min=min_diag, max=max_diag)
        diag_vals.mul_(1 + damping)

        # Batch inverse of 3×3 blocks
        V_inv = torch.linalg.inv(V_blocks)

        return V_inv


class SchurLM(LM):
    """Levenberg-Marquardt with Schur complement for Bundle Adjustment.

    Thin wrapper around ``LM`` with ``method="schur"`` pre-set. Exploits the
    sparse block structure of BA problems by eliminating 3D points to solve a
    much smaller camera-only system.

    All arguments are forwarded to :class:`LM`. See its docstring for details.

    Examples
    --------
    >>> from bae.optim import SchurLM
    >>> from bae.utils.pysolvers import PCG
    >>> optimizer = SchurLM(model, solver=PCG(maxiter=100))
    >>> loss = optimizer.step(input_data)
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('method', 'schur')
        super().__init__(*args, **kwargs)
