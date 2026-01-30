import os
import sys
import subprocess
import shutil
import glob as globmodule

# CRITICAL: Windows-specific fixes BEFORE importing torch
if sys.platform == 'win32':
    # Set DISTUTILS_USE_SDK to prevent VC environment warnings
    os.environ['DISTUTILS_USE_SDK'] = '1'

    # Prepend MSVC bin to PATH to ensure MSVC's link.exe is found before Git's
    # This is critical because Git ships with a link.exe that is NOT a linker
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_msvc_bin = os.path.join(script_dir, '.msvc_bin')

    # Use local copy if available
    if os.path.exists(os.path.join(local_msvc_bin, 'link.exe')):
        os.environ['PATH'] = local_msvc_bin + os.pathsep + os.environ.get('PATH', '')
        print(f"[setup.py] Prepended local MSVC bin to PATH: {local_msvc_bin}")
    else:
        # Find MSVC installation - check common paths (vswhere may not find BuildTools)
        vs_paths = [
            r"C:\Program Files\Microsoft Visual Studio\2022\Community",
            r"C:\Program Files\Microsoft Visual Studio\2022\Professional",
            r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
        ]

        # Also try vswhere
        vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        if os.path.exists(vswhere):
            try:
                result = subprocess.run(
                    [vswhere, "-latest", "-products", "*", "-property", "installationPath"],
                    capture_output=True, text=True
                )
                if result.returncode == 0 and result.stdout.strip():
                    vs_paths.insert(0, result.stdout.strip())
            except Exception:
                pass

        for vs_path in vs_paths:
            msvc_base = os.path.join(vs_path, "VC", "Tools", "MSVC")
            if os.path.exists(msvc_base):
                try:
                    versions = sorted(os.listdir(msvc_base))
                    if versions:
                        latest = versions[-1]
                        msvc_bin = os.path.join(msvc_base, latest, "bin", "Hostx64", "x64")
                        if os.path.exists(os.path.join(msvc_bin, "link.exe")):
                            os.environ['PATH'] = msvc_bin + os.pathsep + os.environ.get('PATH', '')
                            print(f"[setup.py] Prepended MSVC bin to PATH: {msvc_bin}")
                            break
                except Exception:
                    continue

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

VERSION = "0.1"


def _get_msvc_linker_path():
    """Find the correct MSVC linker path."""
    if sys.platform != 'win32':
        return None

    # First check for local .msvc_bin copy
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_linker = os.path.join(script_dir, '.msvc_bin', 'link.exe')
    if os.path.exists(local_linker):
        return local_linker

    # Try common VS installation paths (vswhere may not find BuildTools)
    vs_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
    ]

    # Also try vswhere
    try:
        vswhere_path = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        if os.path.exists(vswhere_path):
            result = subprocess.run(
                [vswhere_path, "-latest", "-products", "*", "-property", "installationPath"],
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                vs_paths.insert(0, result.stdout.strip())
    except Exception:
        pass

    for vs_path in vs_paths:
        msvc_base = os.path.join(vs_path, "VC", "Tools", "MSVC")
        if not os.path.exists(msvc_base):
            continue

        try:
            versions = os.listdir(msvc_base)
            if not versions:
                continue

            latest_version = sorted(versions)[-1]
            for host in ["Hostx64", "HostX86"]:
                linker_path = os.path.join(msvc_base, latest_version, "bin", host, "x64", "link.exe")
                if os.path.exists(linker_path):
                    return linker_path
        except Exception:
            continue

    return None


def _patch_msvc_compiler():
    """Monkey-patch MSVCCompiler to use the correct linker on Windows."""
    if sys.platform != 'win32':
        return

    msvc_linker = _get_msvc_linker_path()
    if not msvc_linker:
        print("[setup.py] Warning: Could not find MSVC linker")
        return

    print(f"[setup.py] Found MSVC linker: {msvc_linker}")

    try:
        from setuptools._distutils._msvccompiler import MSVCCompiler
        _original_initialize = MSVCCompiler.initialize
        _original_link = MSVCCompiler.link

        def _patched_initialize(self, plat_name=None):
            _original_initialize(self, plat_name)
            # Override the linker with the correct MSVC linker
            self.linker = msvc_linker
            print(f"[setup.py] Patched MSVCCompiler.linker to: {msvc_linker}")

        def _patched_link(self, target_desc, objects, output_filename, *args, **kwargs):
            # Force linker to be MSVC's link.exe before every link call
            self.linker = msvc_linker
            return _original_link(self, target_desc, objects, output_filename, *args, **kwargs)

        MSVCCompiler.initialize = _patched_initialize
        MSVCCompiler.link = _patched_link
        print(f"[setup.py] MSVC compiler patched")
    except Exception as e:
        print(f"[setup.py] Warning: Could not patch MSVC compiler: {e}")


# Patch MSVC compiler before any compilation happens
_patch_msvc_compiler()


# Check if we should bundle DLLs (for wheel distribution)
BUNDLE_DLLS = os.environ.get("BUNDLE_DLLS", "0").lower() in ("1", "true", "yes", "y")


def _get_cuda_lib_dir():
    """Find CUDA library directory containing shared libraries."""
    if sys.platform == 'win32':
        # Windows: DLLs are in bin directory
        cuda_path = os.environ.get('CUDA_PATH', '')
        if cuda_path:
            bin_dir = os.path.join(cuda_path, 'bin')
            if os.path.exists(bin_dir):
                return bin_dir

        # Check common CUDA installation paths
        cuda_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        ]
        for path in cuda_paths:
            if os.path.exists(path):
                return path
    else:
        # Linux: .so files are in lib64 or lib directory
        cuda_path = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH', ''))
        if cuda_path:
            for lib_dir in ['lib64', 'lib']:
                path = os.path.join(cuda_path, lib_dir)
                if os.path.exists(path):
                    return path

        # Check common CUDA installation paths
        cuda_paths = [
            '/usr/local/cuda/lib64',
            '/usr/local/cuda-13.0/lib64',
            '/usr/local/cuda-12.8/lib64',
            '/usr/local/cuda-12.6/lib64',
        ]
        for path in cuda_paths:
            if os.path.exists(path):
                return path

    return None


def _get_cudss_lib_dir():
    """Find CUDSS library directory."""
    if sys.platform == 'win32':
        if CUDSS_DIR:
            # CUDSS DLLs are in bin/12 or bin/13
            for cuda_ver in ['13', '12']:
                bin_dir = os.path.join(CUDSS_DIR, 'bin', cuda_ver)
                if os.path.exists(bin_dir):
                    return bin_dir
    else:
        # Linux
        if CUDSS_DIR:
            lib_dir = os.path.join(CUDSS_DIR, 'lib')
            if os.path.exists(lib_dir):
                return lib_dir
        # Default paths
        for path in ['/usr/lib/x86_64-linux-gnu/libcudss/12/', '/usr/local/lib']:
            if os.path.exists(path):
                return path

    return None


def _collect_libs():
    """Collect required shared libraries for bundling."""
    if not BUNDLE_DLLS:
        return []

    libs = []

    if sys.platform == 'win32':
        # Windows DLL patterns
        lib_patterns = [
            'cudart64_*.dll',
            'cublas64_*.dll',
            'cublasLt64_*.dll',
            'cusparse64_*.dll',
            'cusolver64_*.dll',
            'cusolverMg64_*.dll',
            'nvrtc64_*.dll',
            'nvrtc-builtins64_*.dll',
        ]
        ext = '*.dll'
    else:
        # Linux .so patterns
        lib_patterns = [
            'libcudart.so*',
            'libcublas.so*',
            'libcublasLt.so*',
            'libcusparse.so*',
            'libcusolver.so*',
            'libnvrtc.so*',
        ]
        ext = '*.so*'

    cuda_lib = _get_cuda_lib_dir()
    if cuda_lib:
        for pattern in lib_patterns:
            matches = globmodule.glob(os.path.join(cuda_lib, pattern))
            libs.extend(matches)
        print(f"[setup.py] Found {len(libs)} CUDA libraries in {cuda_lib}")

    # CUDSS libraries if enabled
    if USE_CUDSS:
        cudss_lib = _get_cudss_lib_dir()
        if cudss_lib:
            cudss_libs = globmodule.glob(os.path.join(cudss_lib, ext))
            libs.extend(cudss_libs)
            print(f"[setup.py] Found {len(cudss_libs)} CUDSS libraries in {cudss_lib}")

    return libs


class BuildExtensionWithDLLBundle(BuildExtension):
    """Custom BuildExtension that bundles CUDA shared libraries."""

    def run(self):
        # Run the normal build first
        super().run()

        # Bundle shared libraries if requested
        if BUNDLE_DLLS:
            self._bundle_libs()

    def _bundle_libs(self):
        """Copy required shared libraries to the package directory."""
        libs = _collect_libs()
        if not libs:
            print("[setup.py] No libraries to bundle")
            return

        # Determine output directory
        build_lib = self.build_lib
        lib_dest = os.path.join(build_lib, 'bae', 'libs')

        os.makedirs(lib_dest, exist_ok=True)

        bundled = []
        for lib_path in libs:
            lib_name = os.path.basename(lib_path)
            dest_path = os.path.join(lib_dest, lib_name)

            # Handle symlinks on Linux
            if os.path.islink(lib_path):
                # Copy the actual file, not the symlink
                real_path = os.path.realpath(lib_path)
                if os.path.exists(real_path) and not os.path.exists(dest_path):
                    shutil.copy2(real_path, dest_path)
                    bundled.append(lib_name)
            elif not os.path.exists(dest_path):
                shutil.copy2(lib_path, dest_path)
                bundled.append(lib_name)

        if bundled:
            print(f"[setup.py] Bundled {len(bundled)} libraries to {lib_dest}")

            # Create __init__.py in libs directory to make it a package
            init_path = os.path.join(lib_dest, '__init__.py')
            if not os.path.exists(init_path):
                with open(init_path, 'w') as f:
                    f.write('# Directory for bundled CUDA libraries\n')

def readme():
    """Read the README.md file for long description"""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "PyTorch implementation of BA"

# Check if CUDSS should be enabled (enabled by default on all platforms)
USE_CUDSS = os.environ.get("USE_CUDSS", "1").lower() in ("1", "true", "yes", "y")
# Optionally get CUDSS directory from environment variable
# Default to standard Windows installation path if not specified
if sys.platform == 'win32':
    _default_cudss_dir = r"C:\Program Files\NVIDIA cuDSS\v0.7"
    if not os.environ.get("CUDSS_DIR") and os.path.exists(_default_cudss_dir):
        CUDSS_DIR = _default_cudss_dir
    else:
        CUDSS_DIR = os.environ.get("CUDSS_DIR", "")
else:
    CUDSS_DIR = os.environ.get("CUDSS_DIR", "")

if __name__ == '__main__':
    # Common extensions
    # Extra compile args to fix CUDA 12.8 + PyTorch cu128 namespace collision
    # PyTorch's compiled_autograd.h has a known Windows+CUDA compilation issue
    # (see https://github.com/pytorch/pytorch/pull/144707)
    # The workaround requires USE_CUDA to be defined to skip problematic code
    cuda_extra_compile_args = {
        'cxx': ['/DUSE_CUDA'] if sys.platform == 'win32' else ['-DUSE_CUDA'],
        'nvcc': [
            '-DUSE_CUDA',  # Required to trigger PyTorch's Windows+CUDA workaround
        ]
    }

    ext_modules = [
        CppExtension(
            'bae.sparse.bsr',
            [os.path.join('bae', 'sparse', 'sparse_op_cpp.cpp')]
        ),
        CUDAExtension(
            'bae.sparse.bsr_cuda',
            [
                os.path.join('bae', 'sparse', 'sparse_op_cuda.cpp'),
                os.path.join('bae', 'sparse', 'sparse_op_cuda_kernel.cu')
            ],
            extra_compile_args=cuda_extra_compile_args,
        ),
        CUDAExtension(
            'bae.sparse.spgemm',
            [os.path.join('bae', 'sparse', 'cusparse_wrapper.cpp')],
            libraries=['cusparse'],
        ),
        CUDAExtension(
            'bae.sparse.conversion',
            [os.path.join('bae', 'sparse', 'sparse_conversion.cu')],
            extra_compile_args=cuda_extra_compile_args,
            libraries=['cusparse'],
        ),
    ]
    
    # Add CUDSS-dependent extension conditionally
    if USE_CUDSS:
        libraries = ['cusolver', 'cusparse', 'cudss']
        cudss_extra_compile_args = {
            'cxx': ['/DUSE_CUDA'] if sys.platform == 'win32' else [],
            'nvcc': [
                '-DUSE_CUDA',
                '-lcusolver',
                '-lcusparse',
                '-lcudss',
            ]
        }

        # Platform-specific paths
        if sys.platform == 'win32':
            # Windows: CUDSS_DIR must be set
            if CUDSS_DIR:
                # CUDSS on Windows has lib files in version-specific subdirs (lib/12, lib/13)
                # Use lib/12 for CUDA 12.x
                cudss_include = [os.path.join(CUDSS_DIR, 'include')]
                cudss_libdir = [os.path.join(CUDSS_DIR, 'lib', '12')]
                cudss_extra_compile_args['nvcc'].extend([
                    f'-I{CUDSS_DIR}\\include',
                    f'-L{CUDSS_DIR}\\lib\\12',
                ])
            else:
                print("[setup.py] WARNING: CUDSS_DIR not set. Set CUDSS_DIR environment variable to CUDSS installation path.")
                cudss_include = []
                cudss_libdir = []
        else:
            # Linux: use default paths or CUDSS_DIR
            if CUDSS_DIR:
                cudss_include = [os.path.join(CUDSS_DIR, 'include')]
                cudss_libdir = [os.path.join(CUDSS_DIR, 'lib')]
                cudss_extra_compile_args['nvcc'].extend([
                    f'-I{CUDSS_DIR}/include',
                    f'-L{CUDSS_DIR}/lib',
                ])
            else:
                cudss_include = ['/usr/include/libcudss/12/']
                cudss_libdir = ['/usr/lib/x86_64-linux-gnu/libcudss/12/']

        ext_modules.append(
            CUDAExtension(
                'bae.sparse.solve',
                [os.path.join('bae', 'sparse', 'sparse_cusolve.cu')],
                libraries=libraries,
                extra_compile_args=cudss_extra_compile_args,
                include_dirs=cudss_include,
                library_dirs=cudss_libdir
            )
        )

    setup(
        name = 'bae',
        version = VERSION,
        description = 'PyTorch implementation of BA',
        long_description = readme(),
        long_description_content_type = "text/markdown",
        python_requires = ">=3.8",
        install_requires=[
            'torch',
            'torchvision',
            'warp-lang',
        ],
        packages=find_packages(exclude=['./ba_example.py', 
                                        './setup.py', 
                                        './README.md',
                                        './data',
                                        './1dsfm_bal',
                                        './bal_data',
                                        './colmap_helpers',
                                        './datapipes',
                                        './examples',
                                        './tests',]),
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtensionWithDLLBundle},
        package_data={
            'bae': ['libs/*.dll', 'libs/*.so*'] if BUNDLE_DLLS else [],
        },
        include_package_data=True
    )
