"""Custom exceptions for BAE library.

This module defines domain-specific exceptions that provide clear error
messages and help with debugging optimization and sparsity issues.
"""


class BAEError(Exception):
    """Base exception for all BAE errors."""
    pass


class BAESparsityError(BAEError):
    """Exception raised for sparse matrix format or operation errors.

    Raised when:
    - Invalid sparse format is used for an operation
    - Sparse matrix dimensions are incompatible
    - Conversion between sparse formats fails
    """
    pass


class BAEOptimizationError(BAEError):
    """Exception raised for optimization-related errors.

    Raised when:
    - Linear solver fails to converge
    - Jacobian computation fails
    - Invalid optimization parameters are provided
    """
    pass


class BAESolverError(BAEOptimizationError):
    """Exception raised when a linear solver fails.

    Raised when:
    - PCG fails to converge within max iterations
    - CuDSS encounters a singular matrix
    - Matrix is not positive definite
    """
    pass


class BAEDeviceError(BAEError):
    """Exception raised for device-related errors.

    Raised when:
    - Tensors are on different devices
    - CUDA operation fails due to device issues
    - GPU memory is insufficient
    """
    pass
