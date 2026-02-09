"""Schur complement and trust region strategies for optimization.

This module provides custom trust region and adaptive damping strategies that
work with sparse block matrices in BSR format.
"""

import pypose as pp


def _compute_jd(J, D):
    """Compute J @ D for Jacobian(s) and update vector(s).

    Supports two modes:
    1. List mode: J is a list of BSR jacobians, D is a list of update vectors.
       Returns sum of J[i] @ D[i] for all i.
    2. Single tensor mode: J is a single CSR/BSR matrix, D is the full update vector.
       Returns J @ D directly.

    Parameters
    ----------
    J : torch.Tensor or list of torch.Tensor
        Single Jacobian matrix (CSR/BSR) or list of Jacobian matrices.
    D : torch.Tensor or list of torch.Tensor
        Single update vector or list of update vectors.

    Returns
    -------
    torch.Tensor
        Result of J @ D computation, with shape (..., 1).
    """
    # Single tensor mode: J is a single sparse matrix
    if not isinstance(J, (list, tuple)):
        result = J @ D.view(-1)
        return result.view(-1, 1)

    # List mode: J and D are lists of tensors
    JD = None
    for i in range(len(D)):
        jd_i = J[i] @ D[i].view(-1)
        if JD is None:
            JD = jd_i
        else:
            JD = JD + jd_i
    return JD.view(-1, 1)


class TrustRegion(pp.optim.strategy.TrustRegion):
    """Trust region strategy for Levenberg-Marquardt optimization.

    Adjusts the trust region radius based on the quality of each step.
    Compatible with both BSR jacobian lists and single CSR jacobian tensors.
    """

    def update(self, pg, last, loss, J, D, R, *args, **kwargs):
        """Update trust region parameters based on step quality.

        Parameters
        ----------
        pg : dict
            Parameter group containing optimization state.
        last : float
            Previous loss value.
        loss : float
            Current loss value.
        J : torch.Tensor or list of torch.Tensor
            Jacobian matrix (single CSR) or list of Jacobian matrices (BSR).
        D : torch.Tensor or list of torch.Tensor
            Update step vector or list of update vectors.
        R : torch.Tensor
            Residual vector.
        """
        # BSR format supports matrix-vector multiplication directly
        JD = _compute_jd(J, D)
        quality = (last - loss) / -((JD).mT @ (2 * R.view_as(JD) + JD)).squeeze()
        pg['radius'] = 1. / pg['damping']
        if quality > pg['high']:
            pg['radius'] = pg['up'] * pg['radius']
            pg['down'] = self.down
        elif quality > pg['low']:
            pg['radius'] = pg['radius']
            pg['down'] = self.down
        else:
            pg['radius'] = pg['radius'] * pg['down']
            pg['down'] = pg['down'] * pg['factor']
        pg['down'] = max(self.min, min(pg['down'], self.max))
        pg['radius'] = max(self.min, min(pg['radius'], self.max))
        pg['damping'] = 1. / pg['radius']


class Adaptive(pp.optim.strategy.Adaptive):
    """Adaptive damping strategy for Levenberg-Marquardt optimization.

    Adjusts the damping parameter based on the quality of each step.
    Compatible with both BSR jacobian lists and single CSR jacobian tensors.
    """

    def update(self, pg, last, loss, J, D, R, *args, **kwargs):
        """Update damping parameter based on step quality.

        Parameters
        ----------
        pg : dict
            Parameter group containing optimization state.
        last : float
            Previous loss value.
        loss : float
            Current loss value.
        J : torch.Tensor or list of torch.Tensor
            Jacobian matrix (single CSR) or list of Jacobian matrices (BSR).
        D : torch.Tensor or list of torch.Tensor
            Update step vector or list of update vectors.
        R : torch.Tensor
            Residual vector.
        """
        # BSR format supports matrix-vector multiplication directly
        JD = _compute_jd(J, D)
        quality = (last - loss) / -((JD).mT @ (2 * R.view_as(JD) + JD)).squeeze()
        if quality > pg['high']:
            pg['damping'] = pg['damping'] * pg['down']
        elif quality > pg['low']:
            pg['damping'] = pg['damping']
        else:
            pg['damping'] = pg['damping'] * pg['up']
        pg['damping'] = max(self.min, min(pg['damping'], self.max))
