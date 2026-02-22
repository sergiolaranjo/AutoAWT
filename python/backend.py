"""Backend selection: CuPy (GPU) or NumPy (CPU).

Auto-detects GPU availability and provides a unified array module (xp).
All computation modules should import from here:
    from backend import xp, to_numpy, to_device, GPU_AVAILABLE
"""

import os
import sys
import numpy as np

GPU_AVAILABLE = False
GPU_NAME = "N/A"
GPU_COMPUTE_CAP = (0, 0)
GPU_MEMORY_MB = 0
xp = np
cp = None

try:
    import cupy as _cp
    # Verify that a GPU is actually usable
    _dev = _cp.cuda.Device(0)
    GPU_COMPUTE_CAP = _dev.compute_capability
    GPU_AVAILABLE = True
    xp = _cp
    cp = _cp

    # Get GPU info
    try:
        mem = _dev.mem_info
        GPU_MEMORY_MB = mem[1] // (1024 * 1024)  # total memory
    except Exception:
        GPU_MEMORY_MB = 0

    try:
        GPU_NAME = _cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    except Exception:
        GPU_NAME = f"CUDA CC {GPU_COMPUTE_CAP[0]}.{GPU_COMPUTE_CAP[1]}"

    # ---- A100/H100 optimizations ----
    # GPU_COMPUTE_CAP is always a tuple (major, minor) from Device.compute_capability
    cc_major = int(GPU_COMPUTE_CAP[0])

    # Enable TF32 for A100+ (compute capability >= 8.0)
    # TF32 gives ~3x speedup for float32 matmul with minimal precision loss
    if cc_major >= 8:
        try:
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
            os.environ.setdefault('NVIDIA_TF32_OVERRIDE', '1')
        except Exception:
            pass

    # Pre-allocate GPU memory pool to avoid fragmentation
    try:
        mempool = _cp.get_default_memory_pool()
        # Reserve 80% of GPU memory for the pool on high-memory GPUs
        if GPU_MEMORY_MB > 16000:  # A100 80GB / H100
            mempool.set_limit(size=int(GPU_MEMORY_MB * 0.8 * 1024 * 1024))
        elif GPU_MEMORY_MB > 8000:  # A100 40GB / V100 32GB
            mempool.set_limit(size=int(GPU_MEMORY_MB * 0.7 * 1024 * 1024))
    except Exception:
        pass

    print(f"GPU backend: {GPU_NAME} ({GPU_MEMORY_MB} MB, CC {GPU_COMPUTE_CAP})")

except Exception:
    GPU_AVAILABLE = False
    xp = np
    cp = None
    print("CPU backend: NumPy")


def to_numpy(arr):
    """Convert array to NumPy (no-op if already NumPy)."""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_device(arr):
    """Convert NumPy array to device array (CuPy or NumPy)."""
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return arr


def gpu_sync():
    """Synchronize GPU stream (no-op on CPU)."""
    if GPU_AVAILABLE:
        cp.cuda.Stream.null.synchronize()


def gpu_info():
    """Return dict with GPU diagnostics."""
    info = {
        'available': GPU_AVAILABLE,
        'name': GPU_NAME,
        'compute_capability': GPU_COMPUTE_CAP,
        'memory_mb': GPU_MEMORY_MB,
    }
    if GPU_AVAILABLE:
        try:
            mempool = cp.get_default_memory_pool()
            info['pool_used_mb'] = mempool.used_bytes() // (1024 * 1024)
            info['pool_total_mb'] = mempool.total_bytes() // (1024 * 1024)

            free, total = cp.cuda.Device(0).mem_info
            info['gpu_free_mb'] = free // (1024 * 1024)
            info['gpu_total_mb'] = total // (1024 * 1024)
        except Exception:
            pass
    return info


def get_scipy_ndimage():
    """Get the appropriate ndimage module (cupyx.scipy or scipy)."""
    if GPU_AVAILABLE:
        try:
            import cupyx.scipy.ndimage as cnd
            return cnd
        except ImportError:
            pass
    from scipy import ndimage
    return ndimage


def get_sparse_module():
    """Get the appropriate sparse matrix module (cupyx.scipy.sparse or scipy.sparse)."""
    if GPU_AVAILABLE:
        try:
            import cupyx.scipy.sparse as csp
            return csp
        except ImportError:
            pass
    import scipy.sparse as sp
    return sp


def get_sparse_linalg():
    """Get the appropriate sparse linalg module."""
    if GPU_AVAILABLE:
        try:
            import cupyx.scipy.sparse.linalg as csla
            return csla
        except ImportError:
            pass
    import scipy.sparse.linalg as sla
    return sla
