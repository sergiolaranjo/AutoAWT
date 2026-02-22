"""Backend selection: CuPy (GPU) or NumPy (CPU).

Auto-detects GPU availability and provides a unified array module (xp).
All computation modules should import from here:
    from backend import xp, to_numpy, to_device, GPU_AVAILABLE
"""

import numpy as np

try:
    import cupy as cp
    # Verify that a GPU is actually usable
    cp.cuda.Device(0).compute_capability
    GPU_AVAILABLE = True
    xp = cp
    print("GPU backend: CuPy (CUDA)")
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
