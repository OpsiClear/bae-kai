"""BAE: Bundle Adjustment in the Eager-mode.

A PyTorch library for 2nd-order sparse optimization, focused on Bundle Adjustment (BA)
and Pose Graph Optimization (PGO).
"""

# Fix DLL/shared library loading for CUDA extensions
# Libraries need to be in the search path before importing extensions
import sys
import os
import logging

_logger = logging.getLogger(__name__)
_package_dir = os.path.dirname(os.path.abspath(__file__))
_bundled_libs = os.path.join(_package_dir, 'libs')

if sys.platform == 'win32':
    # Windows: Use os.add_dll_directory
    # Always add torch's lib directory first (extensions depend on torch DLLs)
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib):
            os.add_dll_directory(torch_lib)
    except (ImportError, OSError) as e:
        _logger.debug(f"Could not add torch lib directory: {e}")

    # Add bundled CUDA libs if present
    if os.path.exists(_bundled_libs):
        try:
            os.add_dll_directory(_bundled_libs)
        except OSError as e:
            _logger.debug(f"Could not add bundled libs directory: {e}")
    else:
        # Fall back to system CUDA installation
        # Add CUDA toolkit bin directory
        cuda_path = os.environ.get('CUDA_PATH', '')
        if cuda_path:
            cuda_bin = os.path.join(cuda_path, 'bin')
            if os.path.exists(cuda_bin):
                try:
                    os.add_dll_directory(cuda_bin)
                except OSError as e:
                    _logger.debug(f"Could not add CUDA bin directory: {e}")

        # Add CUDSS DLL directory
        cudss_dir = os.environ.get('CUDSS_DIR', r"C:\Program Files\NVIDIA cuDSS\v0.7")
        for cuda_ver in ['13', '12']:
            cudss_dll_path = os.path.join(cudss_dir, 'bin', cuda_ver)
            if os.path.exists(cudss_dll_path):
                try:
                    os.add_dll_directory(cudss_dll_path)
                except OSError as e:
                    _logger.debug(f"Could not add CUDSS directory: {e}")
                break

else:
    # Linux: Modify LD_LIBRARY_PATH or use ctypes to preload
    if os.path.exists(_bundled_libs):
        # Add to LD_LIBRARY_PATH for subprocess calls
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if _bundled_libs not in ld_path:
            os.environ['LD_LIBRARY_PATH'] = _bundled_libs + ':' + ld_path if ld_path else _bundled_libs

        # Preload libraries using ctypes
        import ctypes
        import glob
        for lib in sorted(glob.glob(os.path.join(_bundled_libs, '*.so*'))):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                _logger.debug(f"Could not preload library {lib}: {e}")

# Export custom exceptions
from .exceptions import (
    BAEError,
    BAESparsityError,
    BAEOptimizationError,
    BAESolverError,
    BAEDeviceError,
)
