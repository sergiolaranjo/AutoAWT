"""Wall Thickness detection and computation.
Ported from WT.cpp/h and CUDA kernels in WT_kernel.cu/cuh

Optimized with 3 phases:
  Phase 1: CG solver + RK4 integration + trilinear interpolation
  Phase 2: PyAMG preconditioner + Coupled PDE method (Wang et al. 2019)
  Phase 3: GPU-aware sparse solvers via CuPy
"""

import os
import sys
import math
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy import ndimage

from backend import xp, to_numpy, to_device, GPU_AVAILABLE, get_scipy_ndimage
from utils import export_bmp_wt, normalize_float_to_uint16

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import pyamg
    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False
    print("PyAMG not available, using plain CG (install pyamg for 10-50x faster Laplace)")


# ============================================================
# Kernel-equivalent helper functions (from WT_kernel.cuh)
# ============================================================

def query_tex_buffer(buffer, x, y, z, w, h, d):
    """Safe 3D array access with zero boundary.
    Port of queryTexBuffer from WT_kernel.cuh:1-14
    """
    if 0 <= x < w and 0 <= y < h and 0 <= z < d:
        return buffer[z, y, x]
    return 0.0


def inverse_mask_uint16(mask):
    """Invert a uint16 mask: <1 -> 1, else -> 0."""
    return np.where(mask < 1, np.uint16(1), np.uint16(0))


def inverse_mask_float(mask):
    """Invert a float mask: <1 -> 1.0, else -> 0.0."""
    return np.where(mask < 1.0, 1.0, 0.0).astype(np.float32)


def fillup_volume_by_mask(mask, output, set_value, base_value=0):
    """Fill output where mask == base_value with set_value."""
    output[mask == base_value] = set_value


def cutoff_volume(volume, cutoff, set_value):
    """Set voxels <= cutoff to set_value."""
    volume[volume <= cutoff] = set_value


def binarize(volume, cutoff):
    """Binarize volume: >cutoff -> 1.0, else -> 0.0"""
    return np.where(volume > cutoff, 1.0, 0.0).astype(np.float32)


def subtract_by_bool(float_buf, bool_buf):
    """Zero out float_buf where bool_buf > 0."""
    float_buf[bool_buf > 0] = 0.0


def connectivity_filtering(wall_mask, condition, set_value):
    """For wall voxels with any 26-neighbor > 0 in condition, set output to set_value."""
    d, h, w = wall_mask.shape
    output = np.zeros_like(condition, dtype=np.float32)
    neighbor_max = ndimage.maximum_filter(condition, size=3, mode='constant', cval=0.0)
    mask = (wall_mask > 0) & (neighbor_max > 0)
    output[mask] = set_value
    return output


# ============================================================
# Phase 1+2+3: Sparse Laplacian builder + CG/AMG solver
# ============================================================

def _build_laplacian_3d(shape, interior_mask):
    """Build the 3D Laplacian as a sparse matrix for interior voxels.

    The 7-point stencil (center + 6 neighbors) gives:
        L[i,i] = -6, L[i,j] = 1 for each neighbor j

    Only interior voxels (where interior_mask > 0) are unknowns.
    Boundary voxels contribute to the RHS via Dirichlet conditions.

    Args:
        shape: (D, H, W) volume shape
        interior_mask: boolean 3D array, True for unknowns

    Returns:
        A: sparse CSR matrix (N_interior x N_interior)
        interior_indices: flat indices of interior voxels in the volume
        vol_to_eq: mapping from flat volume index to equation index (-1 if not interior)
    """
    D, H, W = shape
    N = D * H * W

    # Map interior voxels to equation indices
    interior_flat = interior_mask.ravel()
    interior_indices = np.where(interior_flat)[0]
    n_interior = len(interior_indices)

    vol_to_eq = np.full(N, -1, dtype=np.int64)
    vol_to_eq[interior_indices] = np.arange(n_interior, dtype=np.int64)

    # 6 neighbor offsets in flat indexing: +/-1 (x), +/-W (y), +/-W*H (z)
    offsets = np.array([1, -1, W, -W, W * H, -W * H], dtype=np.int64)

    # Compute 3D coordinates of interior voxels for boundary checking
    iz = interior_indices // (H * W)
    iy = (interior_indices % (H * W)) // W
    ix = interior_indices % W

    # Build sparse matrix using COO format
    rows = []
    cols = []
    vals = []

    # Diagonal: start at 0, decrement for each valid neighbor direction
    diag_val = np.zeros(n_interior, dtype=np.float64)

    for offset_idx, offset in enumerate(offsets):
        neighbor_flat = interior_indices + offset

        # Bounds checking per axis
        if offset == 1:
            valid = ix < W - 1
        elif offset == -1:
            valid = ix > 0
        elif offset == W:
            valid = iy < H - 1
        elif offset == -W:
            valid = iy > 0
        elif offset == W * H:
            valid = iz < D - 1
        else:  # -W*H
            valid = iz > 0

        valid_indices = np.where(valid)[0]
        valid_neighbors = neighbor_flat[valid_indices]

        # Each valid neighbor contributes -1 to the diagonal
        diag_val[valid_indices] -= 1.0

        # Check if neighbor is also interior
        neighbor_eq = vol_to_eq[valid_neighbors]
        is_interior_neighbor = neighbor_eq >= 0

        # Interior-interior connections
        int_mask = is_interior_neighbor
        if np.any(int_mask):
            r = valid_indices[int_mask]
            c = neighbor_eq[int_mask]
            rows.append(r)
            cols.append(c)
            vals.append(np.ones(np.sum(int_mask), dtype=np.float64))

    # Add diagonal
    rows.append(np.arange(n_interior, dtype=np.int64))
    cols.append(np.arange(n_interior, dtype=np.int64))
    vals.append(diag_val)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n_interior, n_interior))
    return A, interior_indices, vol_to_eq


def _build_rhs(in_vol, shape, interior_indices, vol_to_eq):
    """Build the RHS vector for the Laplace equation with Dirichlet BCs.

    For each interior voxel, the RHS accounts for known boundary neighbor values:
        b[i] = -sum(boundary_neighbor_values)

    Args:
        in_vol: flat float array with boundary values set (non-interior voxels)
        shape: (D, H, W)
        interior_indices: flat indices of interior voxels
        vol_to_eq: volume-to-equation mapping

    Returns:
        b: RHS vector (n_interior,)
    """
    D, H, W = shape
    n_interior = len(interior_indices)
    b = np.zeros(n_interior, dtype=np.float64)

    iz = interior_indices // (H * W)
    iy = (interior_indices % (H * W)) // W
    ix = interior_indices % W

    offsets = np.array([1, -1, W, -W, W * H, -W * H], dtype=np.int64)

    for offset_idx, offset in enumerate(offsets):
        neighbor_flat = interior_indices + offset

        if offset == 1:
            valid = ix < W - 1
        elif offset == -1:
            valid = ix > 0
        elif offset == W:
            valid = iy < H - 1
        elif offset == -W:
            valid = iy > 0
        elif offset == W * H:
            valid = iz < D - 1
        else:
            valid = iz > 0

        valid_indices = np.where(valid)[0]
        valid_neighbors = neighbor_flat[valid_indices]

        # Boundary neighbors contribute to RHS
        neighbor_eq = vol_to_eq[valid_neighbors]
        is_boundary = neighbor_eq < 0

        if np.any(is_boundary):
            bnd_idx = valid_indices[is_boundary]
            bnd_neighbors = valid_neighbors[is_boundary]
            b[bnd_idx] -= in_vol[bnd_neighbors]

    return b


def _solve_sparse_gpu(A_csr, b_vec, method='cg', tol=1e-5, maxiter=500):
    """Solve sparse system on GPU using CuPy.

    Only used for symmetric positive-definite systems (method='cg').
    Non-symmetric systems (advection operator) should use CPU BiCGSTAB
    for reliable convergence.

    Args:
        A_csr: scipy CSR matrix
        b_vec: numpy RHS vector
        method: 'cg' only (symmetric systems)
        tol: convergence tolerance
        maxiter: maximum iterations

    Returns:
        (x, info) or None if GPU solve fails or method not suitable
    """
    if not GPU_AVAILABLE:
        return None

    # Only use GPU for symmetric CG — non-symmetric systems (bicgstab/gmres)
    # don't converge reliably on GPU with CuPy's iterative solvers
    if method != 'cg':
        return None

    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        import cupyx.scipy.sparse.linalg as csla

        A_f64 = A_csr.astype(np.float64)
        b_f64 = b_vec.astype(np.float64)

        A_gpu = csp.csr_matrix(A_f64)
        b_gpu = cp.asarray(b_f64)

        t0 = time.time()
        x_gpu, info = csla.cg(A_gpu, b_gpu, tol=tol, maxiter=maxiter)
        dt = time.time() - t0

        # Verify convergence with residual check
        residual = b_gpu - A_gpu.dot(x_gpu)
        rel_residual = float(cp.linalg.norm(residual) / max(cp.linalg.norm(b_gpu), 1e-12))

        print(f"  GPU sparse CG: {dt:.2f}s (info={info}, rel_residual={rel_residual:.2e})", file=sys.stderr)

        result_cpu = cp.asnumpy(x_gpu)
        del A_gpu, b_gpu, x_gpu, residual

        # Reject if residual too large (solver didn't converge)
        if rel_residual > 0.01:
            print(f"  GPU residual too large ({rel_residual:.2e}), falling back to CPU", file=sys.stderr)
            return None

        return result_cpu, info

    except Exception as e:
        print(f"  GPU sparse solve failed: {e}", file=sys.stderr)
        return None


def _solve_with_amg_cg(A, b, tol=1e-5, maxiter=500):
    """Solve Ax=b using CG with optional AMG preconditioner (Phase 2).

    Falls back to plain CG if PyAMG is not available.

    Args:
        A: sparse CSR matrix
        b: RHS vector
        tol: convergence tolerance
        maxiter: maximum iterations

    Returns:
        x: solution vector
        info: convergence info (0 = success)
    """
    # Phase 3: GPU sparse CG
    result = _solve_sparse_gpu(A, b, method='cg', tol=tol, maxiter=maxiter)
    if result is not None:
        return result

    # Phase 2: AMG-preconditioned CG (CPU)
    if HAS_PYAMG:
        try:
            ml = pyamg.ruge_stuben_solver(A)
            M = ml.aspreconditioner()
            x, info = sla.cg(A, b, M=M, rtol=tol, maxiter=maxiter)
            return x, info
        except Exception:
            pass

    # Phase 1 fallback: plain CG
    x, info = sla.cg(A, b, rtol=tol, maxiter=maxiter)
    return x, info


def compute_laplace_equation(in_vol, wall_mask, iterations):
    """Solve Laplace equation with boundary conditions using CG solver.

    Replaces the original Jacobi iteration with AMG-preconditioned Conjugate Gradient.
    The wall_mask acts as a multiplicative mask: voxels where mask==0 are held at 0.

    Args:
        in_vol: float32 3D input volume (boundary values set)
        wall_mask: uint16 3D mask (interior region where solution is computed)
        iterations: max iterations (used as CG maxiter)

    Returns:
        result: float32 3D array
    """
    t0 = time.time()
    shape = in_vol.shape
    D, H, W = shape

    # Interior voxels: where wall_mask allows updates (mask > 0)
    # AND where in_vol is 0 (not a fixed boundary value)
    mask_bool = wall_mask.astype(bool)
    boundary_bool = (in_vol != 0) & mask_bool
    interior_bool = mask_bool & ~boundary_bool

    # If nothing to solve, return input
    if not np.any(interior_bool):
        return in_vol.copy()

    # Build sparse system
    A, interior_indices, vol_to_eq = _build_laplacian_3d(shape, interior_bool)
    b = _build_rhs(in_vol.ravel().astype(np.float64), shape, interior_indices, vol_to_eq)

    # Solve
    x, info = _solve_with_amg_cg(A, b, tol=1e-5, maxiter=max(iterations, 500))

    # Reconstruct volume
    result = in_vol.copy().astype(np.float32)
    result_flat = result.ravel()
    result_flat[interior_indices] = x.astype(np.float32)

    # Enforce mask: zero outside wall
    result[~mask_bool] = 0.0

    t1 = time.time()
    print(f"Laplace CG solved in {t1-t0:.2f}s (info={info})", file=sys.stderr)

    return result


def compute_laplace_with_vector(in_vol, G_field, wall_mask, iterations):
    """Solve Laplace equation and compute gradient vector field using CG solver.

    Replaces the original Jacobi iteration. The gradient is computed once
    after convergence instead of every iteration.

    Args:
        in_vol: float32 3D input volume
        G_field: float32 4D array (D, H, W, 4) - gradient + w component
        wall_mask: uint16 3D mask
        iterations: max iterations

    Returns:
        result: float32 3D array (solved potential field)
        G_field: modified gradient field with normalized gradients
    """
    t0 = time.time()
    shape = in_vol.shape
    D, H, W = shape

    # For this version, there's no mask multiplication (original code doesn't mask).
    # Interior = voxels with in_vol == 0.5 (the wall region set by VFInit).
    # Boundaries = voxels with in_vol == 0.0 (chamber/endo) or 1.0 (exterior/epi).
    # We solve for the 0.5-initialized voxels.

    boundary_bool = (in_vol != 0.5) & (in_vol >= 0.0)
    interior_bool = (in_vol == 0.5)

    if not np.any(interior_bool):
        # Fallback: nothing to solve
        return in_vol.copy(), G_field

    # Build sparse system
    A, interior_indices, vol_to_eq = _build_laplacian_3d(shape, interior_bool)
    b = _build_rhs(in_vol.ravel().astype(np.float64), shape, interior_indices, vol_to_eq)

    # Solve
    x, info = _solve_with_amg_cg(A, b, tol=1e-5, maxiter=max(iterations, 500))

    # Reconstruct volume
    result = in_vol.copy().astype(np.float32)
    result_flat = result.ravel()
    result_flat[interior_indices] = x.astype(np.float32)

    t1 = time.time()
    print(f"Laplace+Vector CG solved in {t1-t0:.2f}s (info={info})", file=sys.stderr)

    # Compute gradient field ONCE (central differences) — GPU-accelerated if available
    _xp = xp
    result_d = to_device(result)

    dx = _xp.zeros_like(result_d)
    dy = _xp.zeros_like(result_d)
    dz = _xp.zeros_like(result_d)

    dx[:, :, 1:-1] = (result_d[:, :, 2:] - result_d[:, :, :-2]) * 0.5
    dy[:, 1:-1, :] = (result_d[:, 2:, :] - result_d[:, :-2, :]) * 0.5
    dz[1:-1, :, :] = (result_d[2:, :, :] - result_d[:-2, :, :]) * 0.5

    grad_mag = _xp.sqrt(dx * dx + dy * dy + dz * dz)

    # Normalize gradient and store in G_field
    G_field_d = to_device(G_field)
    valid = grad_mag > 0
    if _xp.any(valid):
        inv_mag = 1.0 / grad_mag[valid]
        G_field_d[valid, 0] = (dx[valid] * inv_mag).astype(_xp.float32)
        G_field_d[valid, 1] = (dy[valid] * inv_mag).astype(_xp.float32)
        G_field_d[valid, 2] = (dz[valid] * inv_mag).astype(_xp.float32)

    G_field = to_numpy(G_field_d)
    del result_d, dx, dy, dz, grad_mag, G_field_d

    return result, G_field


def cuda_ccl(volume, degree_of_connectivity=4):
    """Connected Component Labeling - keep only the largest component."""
    binary = (volume >= 1.0).astype(np.int32)
    struct = ndimage.generate_binary_structure(3, 1)
    labeled, num_features = ndimage.label(binary, structure=struct)

    if num_features == 0:
        return np.ones_like(volume, dtype=np.float32)

    component_sizes = np.bincount(labeled.ravel())
    if len(component_sizes) > 1:
        component_sizes[0] = 0
        largest_label = np.argmax(component_sizes)
    else:
        return np.ones_like(volume, dtype=np.float32)

    print(f"CCL1 min/max:0, {int(np.max(component_sizes))}")
    print(f"CCL argmax:{largest_label}")

    result_uint = np.where(labeled == largest_label, 1, 0).astype(np.uint32)
    result = np.where(result_uint > 0, 0.0, 1.0).astype(np.float32)
    return result


# ============================================================
# Phase 2: Coupled PDE method (Wang et al. 2019)
# Eliminates streamline tracing entirely
# ============================================================

def _build_advection_operator(grad_phi, interior_mask, shape, voxel_size):
    """Build the sparse advection operator for n . grad(T) = 1.

    grad_phi should be the unit normal field n = ∇φ/|∇φ| (dimensionless).
    Uses first-order upwind finite differences for stability:
        If n_x > 0: dT/dx ≈ (T[i] - T[i-1]) / dx  (backward difference)
        If n_x < 0: dT/dx ≈ (T[i+1] - T[i]) / dx  (forward difference)

    Args:
        grad_phi: (D, H, W, 3) unit normal field (normalized gradient of phi)
        interior_mask: boolean 3D array of unknowns
        shape: (D, H, W)
        voxel_size: [sx, sy, sz] voxel spacing

    Returns:
        A: sparse CSR matrix
        interior_indices: flat indices of interior voxels
    """
    D, H, W = shape
    N = D * H * W

    interior_flat = interior_mask.ravel()
    interior_indices = np.where(interior_flat)[0]
    n_interior = len(interior_indices)

    vol_to_eq = np.full(N, -1, dtype=np.int64)
    vol_to_eq[interior_indices] = np.arange(n_interior, dtype=np.int64)

    iz = interior_indices // (H * W)
    iy = (interior_indices % (H * W)) // W
    ix = interior_indices % W

    # Get gradient components at interior voxels
    gx = grad_phi[:, :, :, 0].ravel()[interior_indices]
    gy = grad_phi[:, :, :, 1].ravel()[interior_indices]
    gz = grad_phi[:, :, :, 2].ravel()[interior_indices]

    sx, sy, sz = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])

    rows = []
    cols = []
    vals = []

    # Diagonal contribution from upwind scheme
    diag = np.zeros(n_interior, dtype=np.float64)

    # X-direction: upwind based on sign of gx
    # gx > 0 → backward: gx * (T[i] - T[i-1]) / sx → coeff: gx/sx on diag, -gx/sx on i-1
    # gx < 0 → forward:  gx * (T[i+1] - T[i]) / sx → coeff: -gx/sx on diag, gx/sx on i+1

    # Backward (gx > 0)
    mask_bk_x = (gx > 0) & (ix > 0)
    if np.any(mask_bk_x):
        idx = np.where(mask_bk_x)[0]
        nbr = interior_indices[idx] - 1  # i-1 in x
        nbr_eq = vol_to_eq[nbr]
        is_int = nbr_eq >= 0
        diag[idx] += gx[idx] / sx
        int_idx = idx[is_int]
        rows.append(int_idx)
        cols.append(nbr_eq[is_int])
        vals.append(-gx[int_idx] / sx)

    # Forward (gx < 0)
    mask_fw_x = (gx < 0) & (ix < W - 1)
    if np.any(mask_fw_x):
        idx = np.where(mask_fw_x)[0]
        nbr = interior_indices[idx] + 1  # i+1 in x
        nbr_eq = vol_to_eq[nbr]
        is_int = nbr_eq >= 0
        diag[idx] -= gx[idx] / sx  # -gx is positive when gx < 0
        int_idx = idx[is_int]
        rows.append(int_idx)
        cols.append(nbr_eq[is_int])
        vals.append(gx[int_idx] / sx)

    # Y-direction
    mask_bk_y = (gy > 0) & (iy > 0)
    if np.any(mask_bk_y):
        idx = np.where(mask_bk_y)[0]
        nbr = interior_indices[idx] - W
        nbr_eq = vol_to_eq[nbr]
        is_int = nbr_eq >= 0
        diag[idx] += gy[idx] / sy
        int_idx = idx[is_int]
        rows.append(int_idx)
        cols.append(nbr_eq[is_int])
        vals.append(-gy[int_idx] / sy)

    mask_fw_y = (gy < 0) & (iy < H - 1)
    if np.any(mask_fw_y):
        idx = np.where(mask_fw_y)[0]
        nbr = interior_indices[idx] + W
        nbr_eq = vol_to_eq[nbr]
        is_int = nbr_eq >= 0
        diag[idx] -= gy[idx] / sy
        int_idx = idx[is_int]
        rows.append(int_idx)
        cols.append(nbr_eq[is_int])
        vals.append(gy[int_idx] / sy)

    # Z-direction
    mask_bk_z = (gz > 0) & (iz > 0)
    if np.any(mask_bk_z):
        idx = np.where(mask_bk_z)[0]
        nbr = interior_indices[idx] - W * H
        nbr_eq = vol_to_eq[nbr]
        is_int = nbr_eq >= 0
        diag[idx] += gz[idx] / sz
        int_idx = idx[is_int]
        rows.append(int_idx)
        cols.append(nbr_eq[is_int])
        vals.append(-gz[int_idx] / sz)

    mask_fw_z = (gz < 0) & (iz < D - 1)
    if np.any(mask_fw_z):
        idx = np.where(mask_fw_z)[0]
        nbr = interior_indices[idx] + W * H
        nbr_eq = vol_to_eq[nbr]
        is_int = nbr_eq >= 0
        diag[idx] -= gz[idx] / sz
        int_idx = idx[is_int]
        rows.append(int_idx)
        cols.append(nbr_eq[is_int])
        vals.append(gz[int_idx] / sz)

    # Add diagonal
    rows.append(np.arange(n_interior, dtype=np.int64))
    cols.append(np.arange(n_interior, dtype=np.int64))
    vals.append(diag)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n_interior, n_interior))
    return A, interior_indices


def compute_thickness_coupled_pde(phi, vectorfields, wall_mask, voxel_size):
    """Compute wall thickness using the Coupled PDE method (Wang et al. 2019).

    Solves two first-order PDEs:
        grad(phi) . grad(T_endo) = 1  with T_endo = 0 at endo surface
        grad(phi) . grad(T_epi)  = 1  with T_epi  = 0 at epi surface

    Wall thickness at any point = T_endo + T_epi (in physical units).

    Args:
        phi: float32 3D array - solved Laplace potential (endo=0, epi=1)
        vectorfields: float32 3D array - zone labels (1=endo, 2=wall, 3=epi)
        wall_mask: uint16 3D mask
        voxel_size: [sx, sy, sz] voxel spacing

    Returns:
        endo_vertices: Nx4 array (x, y, z, thickness) for endo surface voxels
    """
    t0 = time.time()
    shape = phi.shape
    D, H, W = shape
    sx, sy, sz = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])

    # Compute gradient of phi in PHYSICAL units (mm) — use GPU if available
    _xp = xp  # CuPy on GPU, NumPy on CPU
    phi_d = to_device(phi)

    grad_phi_d = _xp.zeros((*shape, 3), dtype=_xp.float32)
    # Central differences for interior
    grad_phi_d[:, :, 1:-1, 0] = (phi_d[:, :, 2:] - phi_d[:, :, :-2]) * (0.5 / sx)
    grad_phi_d[:, 1:-1, :, 1] = (phi_d[:, 2:, :] - phi_d[:, :-2, :]) * (0.5 / sy)
    grad_phi_d[1:-1, :, :, 2] = (phi_d[2:, :, :] - phi_d[:-2, :, :]) * (0.5 / sz)
    # Forward/backward differences at boundaries
    if shape[2] > 1:
        grad_phi_d[:, :, 0, 0] = (phi_d[:, :, 1] - phi_d[:, :, 0]) / sx
        grad_phi_d[:, :, -1, 0] = (phi_d[:, :, -1] - phi_d[:, :, -2]) / sx
    if shape[1] > 1:
        grad_phi_d[:, 0, :, 1] = (phi_d[:, 1, :] - phi_d[:, 0, :]) / sy
        grad_phi_d[:, -1, :, 1] = (phi_d[:, -1, :] - phi_d[:, -2, :]) / sy
    if shape[0] > 1:
        grad_phi_d[0, :, :, 2] = (phi_d[1, :, :] - phi_d[0, :, :]) / sz
        grad_phi_d[-1, :, :, 2] = (phi_d[-1, :, :] - phi_d[-2, :, :]) / sz

    # Normalize to unit normal: n = grad(phi) / |grad(phi)|
    mag = _xp.sqrt(grad_phi_d[:, :, :, 0]**2 + grad_phi_d[:, :, :, 1]**2 +
                   grad_phi_d[:, :, :, 2]**2)
    mag = _xp.maximum(mag, 1e-12)
    grad_phi_d[:, :, :, 0] /= mag
    grad_phi_d[:, :, :, 1] /= mag
    grad_phi_d[:, :, :, 2] /= mag

    grad_phi = to_numpy(grad_phi_d)
    del phi_d, grad_phi_d, mag

    # Identify regions
    endo_surface = (vectorfields == 1.0)
    epi_surface = (vectorfields == 3.0)
    wall_region = (wall_mask > 0)

    # Interior for T_endo: wall voxels that are NOT on the endo surface
    interior_endo = wall_region & ~endo_surface & ~epi_surface
    # Interior for T_epi: wall voxels that are NOT on the epi surface
    interior_epi = wall_region & ~epi_surface & ~endo_surface

    # Build and solve T_endo: grad(phi) . grad(T_endo) = 1, T_endo=0 at endo
    A_endo, idx_endo = _build_advection_operator(grad_phi, interior_endo, shape, voxel_size)
    b_endo = np.ones(len(idx_endo), dtype=np.float64)

    # Regularize: add small diagonal to avoid singular matrix
    A_endo = A_endo + sp.eye(A_endo.shape[0], format='csr') * 1e-6

    n_unknowns = len(idx_endo)
    print(f"Solving coupled PDE for T_endo ({n_unknowns} unknowns) [CPU BiCGSTAB]...", file=sys.stderr)

    # Use CPU BiCGSTAB for non-symmetric advection operator
    # (GPU iterative solvers don't converge reliably for this system)
    t_solve = time.time()
    T_endo_vals, info_endo = sla.bicgstab(A_endo, b_endo, rtol=1e-4, maxiter=1000)
    dt_solve = time.time() - t_solve
    residual_endo = np.linalg.norm(A_endo @ T_endo_vals - b_endo) / max(np.linalg.norm(b_endo), 1e-12)
    print(f"  T_endo: info={info_endo}, time={dt_solve:.1f}s, rel_residual={residual_endo:.2e}, "
          f"min={np.min(T_endo_vals):.4f}, max={np.max(T_endo_vals):.4f}, mean={np.mean(np.abs(T_endo_vals)):.4f}",
          file=sys.stderr)
    if info_endo != 0:
        print(f"  T_endo BiCGSTAB did not converge, trying spsolve", file=sys.stderr)
        try:
            T_endo_vals = sla.spsolve(A_endo, b_endo)
        except Exception:
            T_endo_vals = np.zeros(n_unknowns, dtype=np.float64)

    # Reconstruct T_endo volume
    T_endo = np.zeros(D * H * W, dtype=np.float32)
    T_endo[idx_endo] = np.abs(T_endo_vals).astype(np.float32)
    T_endo = T_endo.reshape(shape)

    # Build T_epi: solve with gradient reversed (flow from epi to endo)
    # Negate gradient to reverse the direction
    grad_phi_rev = -grad_phi
    A_epi, idx_epi = _build_advection_operator(grad_phi_rev, interior_epi, shape, voxel_size)
    A_epi = A_epi + sp.eye(A_epi.shape[0], format='csr') * 1e-6
    b_epi = np.ones(len(idx_epi), dtype=np.float64)

    print(f"Solving coupled PDE for T_epi ({len(idx_epi)} unknowns) [CPU BiCGSTAB]...", file=sys.stderr)
    t_solve = time.time()
    T_epi_vals, info_epi = sla.bicgstab(A_epi, b_epi, rtol=1e-4, maxiter=1000)
    dt_solve = time.time() - t_solve
    residual_epi = np.linalg.norm(A_epi @ T_epi_vals - b_epi) / max(np.linalg.norm(b_epi), 1e-12)
    print(f"  T_epi: info={info_epi}, time={dt_solve:.1f}s, rel_residual={residual_epi:.2e}, "
          f"min={np.min(T_epi_vals):.4f}, max={np.max(T_epi_vals):.4f}, mean={np.mean(np.abs(T_epi_vals)):.4f}",
          file=sys.stderr)
    if info_epi != 0:
        print(f"  T_epi BiCGSTAB did not converge, trying spsolve", file=sys.stderr)
        try:
            T_epi_vals = sla.spsolve(A_epi, b_epi)
        except Exception:
            T_epi_vals = np.zeros(len(idx_epi), dtype=np.float64)

    T_epi = np.zeros(D * H * W, dtype=np.float32)
    T_epi[idx_epi] = np.abs(T_epi_vals).astype(np.float32)
    T_epi = T_epi.reshape(shape)

    # Total thickness at each wall voxel
    thickness_field = T_endo + T_epi

    # Extract endo surface vertices with their thickness
    endo_coords = np.argwhere(endo_surface)  # (N, 3) = (z, y, x)
    if len(endo_coords) > 0:
        endo_vertices = np.zeros((len(endo_coords), 4), dtype=np.float32)
        endo_vertices[:, 0] = endo_coords[:, 2].astype(np.float32)  # x
        endo_vertices[:, 1] = endo_coords[:, 1].astype(np.float32)  # y
        endo_vertices[:, 2] = endo_coords[:, 0].astype(np.float32)  # z

        # Sample thickness: use max of 6-neighbors (vectorized with maximum_filter)
        max_thickness = ndimage.maximum_filter(thickness_field, size=3, mode='constant', cval=0.0)
        endo_vertices[:, 3] = max_thickness[endo_coords[:, 0], endo_coords[:, 1], endo_coords[:, 2]]
    else:
        endo_vertices = np.zeros((0, 4), dtype=np.float32)

    t1 = time.time()
    wt_vals = endo_vertices[:, 3] if len(endo_vertices) > 0 else np.array([0.0])
    wt_pos = wt_vals[wt_vals > 0]
    print(f"Coupled PDE thickness computed in {t1-t0:.2f}s", file=sys.stderr)
    print(f"  Thickness: mean={np.mean(wt_pos):.3f}mm, median={np.median(wt_pos):.3f}mm, "
          f"max={np.max(wt_pos):.3f}mm, points={len(wt_pos)}", file=sys.stderr)

    return endo_vertices


# ============================================================
# Phase 1: RK4 + Trilinear Interpolation (fallback streamline method)
# ============================================================

def _trilinear_interp(field, px, py, pz, w, h, d):
    """Trilinear interpolation of a 4D vector field at position (px, py, pz).
    Returns (vf0, vf1, vf2, vf3). Pure Python for Numba compatibility.
    """
    # Clamp to valid range
    fx = max(0.0, min(px, w - 1.001))
    fy = max(0.0, min(py, h - 1.001))
    fz = max(0.0, min(pz, d - 1.001))

    x0 = int(fx)
    y0 = int(fy)
    z0 = int(fz)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    z1 = min(z0 + 1, d - 1)

    xd = fx - x0
    yd = fy - y0
    zd = fz - z0

    result = [0.0, 0.0, 0.0, 0.0]
    for c in range(4):
        # Interpolate along x
        c00 = field[z0, y0, x0, c] * (1 - xd) + field[z0, y0, x1, c] * xd
        c01 = field[z1, y0, x0, c] * (1 - xd) + field[z1, y0, x1, c] * xd
        c10 = field[z0, y1, x0, c] * (1 - xd) + field[z0, y1, x1, c] * xd
        c11 = field[z1, y1, x0, c] * (1 - xd) + field[z1, y1, x1, c] * xd
        # Interpolate along y
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        # Interpolate along z
        result[c] = c0 * (1 - zd) + c1 * zd

    return result[0], result[1], result[2], result[3]


def _compute_thickness_single_rk4(start_x, start_y, start_z,
                                   vector_fields, wall_mask,
                                   w, h, d, voxel_size, mode_sign):
    """Compute thickness for a single vertex using RK4 + trilinear interpolation.

    Phase 1 optimization: dt=0.05 with RK4 (vs dt=0.001 with Euler).
    ~25x faster per vertex due to 100x fewer steps, 4x more work per step.
    """
    MAX_TRAVEL = 256
    dt = 0.05
    max_steps = int(MAX_TRAVEL / dt)

    sx = int(start_x)
    sy = int(start_y)
    sz = int(start_z)

    if sx >= w or sy >= h or sz >= d or sx < 0 or sy < 0 or sz < 0:
        return 0.0

    px = start_x + 0.5
    py = start_y + 0.5
    pz = start_z + 0.5

    prev_dx = 0.0
    prev_dy = 0.0
    prev_dz = 0.0
    first_sample = True
    state = True
    thickness = 0.0

    for step in range(max_steps):
        ix = int(px)
        iy = int(py)
        iz = int(pz)

        if ix >= w or iy >= h or iz >= d or ix < 0 or iy < 0 or iz < 0:
            break

        vf0, vf1, vf2, vf3 = _trilinear_interp(vector_fields, px, py, pz, w, h, d)

        if vf3 > 0:
            state = False
            # Initialize prev direction from first gradient sample
            if first_sample:
                prev_dx = vf0 * mode_sign
                prev_dy = vf1 * mode_sign
                prev_dz = vf2 * mode_sign
                first_sample = False

            dot_val = prev_dx * vf0 + prev_dy * vf1 + prev_dz * vf2

            if dot_val >= 0:
                cur_dx = vf0 * mode_sign
                cur_dy = vf1 * mode_sign
                cur_dz = vf2 * mode_sign
                dir_len = math.sqrt(cur_dx * cur_dx + cur_dy * cur_dy + cur_dz * cur_dz)

                if dir_len > 0:
                    prev_px = px
                    prev_py = py
                    prev_pz = pz

                    # RK4 integration
                    k1x = cur_dx * dt
                    k1y = cur_dy * dt
                    k1z = cur_dz * dt

                    mx = px + k1x * 0.5
                    my = py + k1y * 0.5
                    mz = pz + k1z * 0.5
                    if 0 <= mx < w and 0 <= my < h and 0 <= mz < d:
                        v0, v1, v2, v3 = _trilinear_interp(vector_fields, mx, my, mz, w, h, d)
                        k2x = v0 * mode_sign * dt
                        k2y = v1 * mode_sign * dt
                        k2z = v2 * mode_sign * dt
                    else:
                        k2x, k2y, k2z = k1x, k1y, k1z

                    mx = px + k2x * 0.5
                    my = py + k2y * 0.5
                    mz = pz + k2z * 0.5
                    if 0 <= mx < w and 0 <= my < h and 0 <= mz < d:
                        v0, v1, v2, v3 = _trilinear_interp(vector_fields, mx, my, mz, w, h, d)
                        k3x = v0 * mode_sign * dt
                        k3y = v1 * mode_sign * dt
                        k3z = v2 * mode_sign * dt
                    else:
                        k3x, k3y, k3z = k2x, k2y, k2z

                    mx = px + k3x
                    my = py + k3y
                    mz = pz + k3z
                    if 0 <= mx < w and 0 <= my < h and 0 <= mz < d:
                        v0, v1, v2, v3 = _trilinear_interp(vector_fields, mx, my, mz, w, h, d)
                        k4x = v0 * mode_sign * dt
                        k4y = v1 * mode_sign * dt
                        k4z = v2 * mode_sign * dt
                    else:
                        k4x, k4y, k4z = k3x, k3y, k3z

                    px += (k1x + 2*k2x + 2*k3x + k4x) / 6.0
                    py += (k1y + 2*k2y + 2*k3y + k4y) / 6.0
                    pz += (k1z + 2*k2z + 2*k3z + k4z) / 6.0

                    # Check wall mask at previous position
                    wix = min(max(int(prev_px), 0), w - 1)
                    wiy = min(max(int(prev_py), 0), h - 1)
                    wiz = min(max(int(prev_pz), 0), d - 1)
                    wall_val = wall_mask[wiz, wiy, wix]

                    if wall_val > 0:
                        ddx = (prev_px - px) * voxel_size[0]
                        ddy = (prev_py - py) * voxel_size[1]
                        ddz = (prev_pz - pz) * voxel_size[2]
                        thickness += math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)

                    prev_dx = cur_dx
                    prev_dy = cur_dy
                    prev_dz = cur_dz
                else:
                    break
            else:
                if step == 0:
                    prev_dx = vf0 * mode_sign
                    prev_dy = vf1 * mode_sign
                    prev_dz = vf2 * mode_sign
                    px += prev_dx * dt
                    py += prev_dy * dt
                    pz += prev_dz * dt
                    continue
                break
        else:
            if not state:
                break
            else:
                prev_dx = vf0 * mode_sign
                prev_dy = vf1 * mode_sign
                prev_dz = vf2 * mode_sign
                px += prev_dx * dt
                py += prev_dy * dt
                pz += prev_dz * dt

    return thickness


# ============================================================
# Numba JIT compilation for RK4 thickness
# ============================================================

if HAS_NUMBA:
    _trilinear_interp_jit = njit(_trilinear_interp, cache=True)
    _thickness_rk4_jit = njit(_compute_thickness_single_rk4, cache=True)

    @njit(parallel=True, cache=True)
    def _compute_thickness_batch_rk4(vertices_xyz, vector_fields, wall_mask,
                                      w, h, d, voxel_size, mode_sign):
        """Batch compute thickness using RK4 + trilinear for all vertices."""
        N = vertices_xyz.shape[0]
        result = np.zeros(N, dtype=np.float32)

        for idx in prange(N):
            result[idx] = _thickness_rk4_jit(
                vertices_xyz[idx, 0], vertices_xyz[idx, 1], vertices_xyz[idx, 2],
                vector_fields, wall_mask,
                w, h, d, voxel_size, mode_sign
            )

        return result
else:
    def _compute_thickness_batch_rk4(vertices_xyz, vector_fields, wall_mask,
                                      w, h, d, voxel_size, mode_sign):
        """Batch compute thickness - pure Python RK4 fallback."""
        N = vertices_xyz.shape[0]
        result = np.zeros(N, dtype=np.float32)

        for idx in range(N):
            result[idx] = _compute_thickness_single_rk4(
                vertices_xyz[idx, 0], vertices_xyz[idx, 1], vertices_xyz[idx, 2],
                vector_fields, wall_mask,
                w, h, d, voxel_size, mode_sign
            )
            if idx % 1000 == 0 and idx > 0:
                print(f"  thickness progress: {idx}/{N}", file=sys.stderr)

        return result


# Phase 3: GPU batch RK4 via CuPy raw kernel (when coupled PDE fallback is needed)
_GPU_RK4_KERNEL = None

if GPU_AVAILABLE:
    try:
        import cupy as cp

        _GPU_RK4_KERNEL_CODE = r"""
// Trilinear interpolation of a 4-component vector field on GPU
__device__ void trilinear_vf(
    const float* vf, int W, int H, int D,
    float px, float py, float pz,
    float &ox, float &oy, float &oz, float &ow
) {
    float fx = fminf(fmaxf(px, 0.0f), (float)(W - 1) - 0.001f);
    float fy = fminf(fmaxf(py, 0.0f), (float)(H - 1) - 0.001f);
    float fz = fminf(fmaxf(pz, 0.0f), (float)(D - 1) - 0.001f);

    int x0 = (int)fx, y0 = (int)fy, z0 = (int)fz;
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, H - 1);
    int z1 = min(z0 + 1, D - 1);

    float xd = fx - x0, yd = fy - y0, zd = fz - z0;

    // 8 corners, 4 components each
    #define VF4(X,Y,Z,C) vf[((Z)*H + (Y))*W*4 + (X)*4 + (C)]

    for (int c = 0; c < 4; c++) {
        float c00 = VF4(x0,y0,z0,c)*(1-xd) + VF4(x1,y0,z0,c)*xd;
        float c01 = VF4(x0,y0,z1,c)*(1-xd) + VF4(x1,y0,z1,c)*xd;
        float c10 = VF4(x0,y1,z0,c)*(1-xd) + VF4(x1,y1,z0,c)*xd;
        float c11 = VF4(x0,y1,z1,c)*(1-xd) + VF4(x1,y1,z1,c)*xd;
        float c0 = c00*(1-yd) + c10*yd;
        float c1 = c01*(1-yd) + c11*yd;
        float val = c0*(1-zd) + c1*zd;
        if (c == 0) ox = val;
        else if (c == 1) oy = val;
        else if (c == 2) oz = val;
        else ow = val;
    }
    #undef VF4
}

extern "C" __global__
void rk4_thickness_kernel(
    const float* vertices,     // Nx3
    const float* vector_fields,// D*H*W*4
    const unsigned short* wall_mask, // D*H*W
    const int W, const int H, const int D,
    const float vsx, const float vsy, const float vsz,
    const float mode_sign,
    float* out_thickness,      // N
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float start_x = vertices[idx * 3 + 0];
    float start_y = vertices[idx * 3 + 1];
    float start_z = vertices[idx * 3 + 2];

    int sx = (int)start_x;
    int sy = (int)start_y;
    int sz = (int)start_z;

    if (sx >= W || sy >= H || sz >= D || sx < 0 || sy < 0 || sz < 0) {
        out_thickness[idx] = 0.0f;
        return;
    }

    float px = start_x + 0.5f;
    float py = start_y + 0.5f;
    float pz = start_z + 0.5f;

    float pdx = 0.0f, pdy = 0.0f, pdz = 0.0f;
    bool state = true;
    float thickness = 0.0f;
    float dt = 0.05f;
    int max_steps = 5120;

    for (int step = 0; step < max_steps; step++) {
        int ix = (int)px;
        int iy = (int)py;
        int iz = (int)pz;

        if (ix >= W || iy >= H || iz >= D || ix < 0 || iy < 0 || iz < 0) break;

        // Trilinear-interpolated vector field at current position
        // Use px,py,pz directly (already offset by +0.5 at start, matching Python)
        float vf0, vf1, vf2, vf3;
        trilinear_vf(vector_fields, W, H, D, px, py, pz,
                     vf0, vf1, vf2, vf3);

        if (vf3 > 0) {
            state = false;
            float dot_val = pdx * vf0 + pdy * vf1 + pdz * vf2;

            if (dot_val >= 0) {
                float dx = vf0 * mode_sign;
                float dy = vf1 * mode_sign;
                float dz = vf2 * mode_sign;
                float dir_len = sqrtf(dx*dx + dy*dy + dz*dz);

                if (dir_len > 0) {
                    float prev_px = px, prev_py = py, prev_pz = pz;

                    // RK4 integration with trilinear interpolation
                    // k1
                    float k1x = dx * dt, k1y = dy * dt, k1z = dz * dt;

                    // k2 — midpoint using k1
                    float mx = px + k1x*0.5f;
                    float my = py + k1y*0.5f;
                    float mz = pz + k1z*0.5f;
                    float m0, m1, m2, m3;
                    trilinear_vf(vector_fields, W, H, D, mx, my, mz, m0, m1, m2, m3);
                    float k2x = m0*mode_sign*dt, k2y = m1*mode_sign*dt, k2z = m2*mode_sign*dt;

                    // k3 — midpoint using k2
                    mx = px + k2x*0.5f;
                    my = py + k2y*0.5f;
                    mz = pz + k2z*0.5f;
                    trilinear_vf(vector_fields, W, H, D, mx, my, mz, m0, m1, m2, m3);
                    float k3x = m0*mode_sign*dt, k3y = m1*mode_sign*dt, k3z = m2*mode_sign*dt;

                    // k4 — endpoint using k3
                    mx = px + k3x;
                    my = py + k3y;
                    mz = pz + k3z;
                    trilinear_vf(vector_fields, W, H, D, mx, my, mz, m0, m1, m2, m3);
                    float k4x = m0*mode_sign*dt, k4y = m1*mode_sign*dt, k4z = m2*mode_sign*dt;

                    // Weighted sum
                    px += (k1x + 2*k2x + 2*k3x + k4x) / 6.0f;
                    py += (k1y + 2*k2y + 2*k3y + k4y) / 6.0f;
                    pz += (k1z + 2*k2z + 2*k3z + k4z) / 6.0f;

                    int wix = min(max((int)prev_px, 0), W - 1);
                    int wiy = min(max((int)prev_py, 0), H - 1);
                    int wiz = min(max((int)prev_pz, 0), D - 1);
                    unsigned short wv = wall_mask[(wiz * H + wiy) * W + wix];

                    if (wv > 0) {
                        float ddx = (prev_px - px) * vsx;
                        float ddy = (prev_py - py) * vsy;
                        float ddz = (prev_pz - pz) * vsz;
                        thickness += sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
                    }

                    pdx = dx; pdy = dy; pdz = dz;
                } else {
                    break;
                }
            } else {
                if (step == 0) {
                    pdx = vf0 * mode_sign;
                    pdy = vf1 * mode_sign;
                    pdz = vf2 * mode_sign;
                    px += pdx * dt;
                    py += pdy * dt;
                    pz += pdz * dt;
                    continue;
                }
                break;
            }
        } else {
            if (!state) break;
            pdx = vf0 * mode_sign;
            pdy = vf1 * mode_sign;
            pdz = vf2 * mode_sign;
            px += pdx * dt;
            py += pdy * dt;
            pz += pdz * dt;
        }
    }

    out_thickness[idx] = thickness;
}
"""
        _GPU_RK4_KERNEL = cp.RawKernel(_GPU_RK4_KERNEL_CODE, 'rk4_thickness_kernel')
    except Exception:
        _GPU_RK4_KERNEL = None


def _compute_thickness_batch_gpu(vertices_xyz, vector_fields, wall_mask,
                                  w, h, d, voxel_size, mode_sign):
    """Phase 3: GPU batch thickness computation using CUDA kernel."""
    import cupy as cp

    N = vertices_xyz.shape[0]
    if N == 0:
        return np.zeros(0, dtype=np.float32)

    d_verts = cp.asarray(vertices_xyz.astype(np.float32))
    d_vf = cp.asarray(vector_fields.astype(np.float32))
    d_mask = cp.asarray(wall_mask.astype(np.uint16))
    d_out = cp.zeros(N, dtype=cp.float32)

    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    _GPU_RK4_KERNEL(
        (grid_size,), (block_size,),
        (d_verts, d_vf, d_mask,
         np.int32(w), np.int32(h), np.int32(d),
         np.float32(voxel_size[0]), np.float32(voxel_size[1]), np.float32(voxel_size[2]),
         np.float32(mode_sign), d_out, np.int32(N))
    )

    return cp.asnumpy(d_out)


# ============================================================
# Main WT class
# ============================================================

class WT:
    """Wall Thickness computation class.

    Optimized with:
      Phase 1: CG solver + RK4 integration + trilinear interpolation
      Phase 2: PyAMG preconditioner + Coupled PDE (eliminates streamlines)
      Phase 3: GPU sparse CG + CUDA batch tracing
    """

    def __init__(self, save_path, volume_size, voxel_spacing, volume_position, wall_mask, convex_mask):
        self.m_save_path = save_path

        if len(volume_size) == 4:
            self.m_volume_size = np.array([volume_size[0], volume_size[1], volume_size[2]], dtype=np.float32)
        else:
            self.m_volume_size = np.array(volume_size[:3], dtype=np.float32)

        self.m_voxel_spacing = voxel_spacing.copy().astype(np.float32)
        self.m_volume_position = volume_position.copy().astype(np.float32)

        self.m_wall_mask = wall_mask.copy().astype(np.uint16)
        self.m_convex_mask = convex_mask.copy().astype(np.uint16)
        self.m_chamber_mask = np.zeros_like(wall_mask, dtype=np.uint16)

        self.m_VFInit = None
        self.m_vectorfields = None

        self.m_endo_vertices_list = []

    def get_chamber_mask(self):
        return self.m_chamber_mask

    def detect_epi_endo(self, main_volume=None):
        """Detect endocardium and epicardium.
        Exact port of WT::detectEpiEndo() from WT.cpp:80-192
        """
        w = int(self.m_volume_size[0])
        h = int(self.m_volume_size[1])
        d = int(self.m_volume_size[2])

        d_vf3D_frontbuf = np.zeros((d, h, w), dtype=np.float32)
        d_vf3D_backbuf = np.zeros((d, h, w), dtype=np.float32)

        self.m_convex_mask = inverse_mask_uint16(self.m_convex_mask)
        # Invert wall_mask for Laplace solve (inverted back after at line below)
        self.m_wall_mask = inverse_mask_uint16(self.m_wall_mask)

        fillup_volume_by_mask(self.m_convex_mask, d_vf3D_frontbuf, 10.0, base_value=1)

        d_vf3D_backbuf = compute_laplace_equation(
            d_vf3D_frontbuf, self.m_wall_mask, 100
        )

        cutoff_volume(d_vf3D_backbuf, 1e-01, -2.0)
        d_vf3D_backbuf = binarize(d_vf3D_backbuf, -2.0)
        d_vf3D_backbuf = inverse_mask_float(d_vf3D_backbuf)

        self.m_wall_mask = inverse_mask_uint16(self.m_wall_mask)
        subtract_by_bool(d_vf3D_backbuf, self.m_wall_mask)

        self.m_convex_mask = None

        d_vf3D_backbuf = cuda_ccl(d_vf3D_backbuf, 4)
        subtract_by_bool(d_vf3D_backbuf, self.m_wall_mask)

        norm_buf = normalize_float_to_uint16(d_vf3D_backbuf)
        self.m_chamber_mask = norm_buf.astype(np.uint16)

        self.m_VFInit = d_vf3D_backbuf.copy()
        fillup_volume_by_mask(self.m_wall_mask, self.m_VFInit, 0.5, base_value=1)

        d_vf3D_frontbuf = np.zeros((d, h, w), dtype=np.float32)
        fillup_volume_by_mask(self.m_wall_mask, d_vf3D_frontbuf, 2.0, base_value=1)

        epi_result = connectivity_filtering(self.m_wall_mask, d_vf3D_backbuf, 3.0)
        d_vf3D_frontbuf = np.where(epi_result > 0, epi_result, d_vf3D_frontbuf)

        d_vf3D_backbuf = inverse_mask_float(d_vf3D_backbuf)
        subtract_by_bool(d_vf3D_backbuf, self.m_wall_mask)
        endo_result = connectivity_filtering(self.m_wall_mask, d_vf3D_backbuf, 1.0)
        d_vf3D_frontbuf = np.where(endo_result > 0, endo_result, d_vf3D_frontbuf)

        self.m_vectorfields = d_vf3D_frontbuf.copy()

        export_bmp_wt(d_vf3D_frontbuf, (w, h, d), self.m_save_path, "Epi-Endo")

    def eval_wt(self):
        """Evaluate wall thickness.

        Uses Coupled PDE method (Phase 2) as primary approach.
        Falls back to RK4 streamline tracing (Phase 1) if coupled PDE fails.
        """
        w = int(self.m_volume_size[0])
        h = int(self.m_volume_size[1])
        d = int(self.m_volume_size[2])

        # Initialize voltage from VFInit
        voltage3D = self.m_VFInit.copy()

        # Vector fields (4 components: xyz gradient + w)
        vector_fields = np.zeros((d, h, w, 4), dtype=np.float32)
        vector_fields[:, :, :, 3] = 1.0

        # Solve Laplace with CG (Phase 1+2+3)
        voltage3D_solved, vector_fields = compute_laplace_with_vector(
            voltage3D, vector_fields, self.m_wall_mask, 400
        )

        h_vectorfields = self.m_vectorfields
        voxel_size = self.m_voxel_spacing.copy()

        # Try Coupled PDE method (Phase 2)
        try:
            print("Using Coupled PDE method (Phase 2)...", file=sys.stderr)
            endo_vertices = compute_thickness_coupled_pde(
                voltage3D_solved, h_vectorfields, self.m_wall_mask, voxel_size
            )

            endo_cnt = len(endo_vertices)
            epi_cnt = int(np.sum(h_vectorfields == 3.0))
            print(f"total_endoVCnt_size = {endo_cnt}, {epi_cnt}", file=sys.stderr)

            if endo_cnt > 0 and np.mean(endo_vertices[:, 3]) > 0:
                print("Coupled PDE succeeded", file=sys.stderr)
            else:
                raise ValueError("Coupled PDE produced zero thickness, falling back to RK4")

        except Exception as e:
            print(f"Coupled PDE failed ({e}), falling back to RK4 streamlines", file=sys.stderr)
            endo_vertices = self._eval_wt_streamline(vector_fields, w, h, d)

        # Save PLT
        self.m_endo_vertices_list = []
        self._save_plt(
            os.path.join(self.m_save_path, "WT-endo"),
            endo_vertices, self.m_endo_vertices_list
        )

    def _eval_wt_streamline(self, vector_fields, w, h, d):
        """Fallback: RK4 streamline tracing (Phase 1 + Phase 3 GPU)."""
        h_vectorfields = self.m_vectorfields

        endo_coords = np.argwhere(h_vectorfields == 1.0)
        if len(endo_coords) > 0:
            endo_vertices = np.zeros((len(endo_coords), 4), dtype=np.float32)
            endo_vertices[:, 0] = endo_coords[:, 2].astype(np.float32)
            endo_vertices[:, 1] = endo_coords[:, 1].astype(np.float32)
            endo_vertices[:, 2] = endo_coords[:, 0].astype(np.float32)
        else:
            endo_vertices = np.zeros((0, 4), dtype=np.float32)

        epi_coords = np.argwhere(h_vectorfields == 3.0)
        endo_cnt = len(endo_vertices)
        epi_cnt = len(epi_coords)
        print(f"total_endoVCnt_size = {endo_cnt}, {epi_cnt}", file=sys.stderr)

        voxel_size = self.m_voxel_spacing.copy()
        endo_vertices = self._compute_thickness(
            endo_vertices, vector_fields, self.m_wall_mask, (w, h, d), voxel_size, mode=0
        )

        return endo_vertices

    def _compute_thickness(self, vertices, vector_fields, wall_mask, vol_size, voxel_size, mode=0):
        """Compute wall thickness using RK4 (Phase 1) or GPU kernel (Phase 3)."""
        w, h, d = vol_size
        mode_sign = np.float32(-1.0 if mode else 1.0)
        vs = voxel_size.astype(np.float32)

        # Phase 3: use GPU kernel if available
        if GPU_AVAILABLE and _GPU_RK4_KERNEL is not None:
            print("  Using GPU CUDA kernel for thickness", file=sys.stderr)
            thicknesses = _compute_thickness_batch_gpu(
                vertices[:, :3].astype(np.float32),
                vector_fields.astype(np.float32),
                wall_mask.astype(np.uint16),
                w, h, d, vs, mode_sign
            )
        else:
            # Phase 1: RK4 + trilinear (CPU, Numba-accelerated)
            thicknesses = _compute_thickness_batch_rk4(
                vertices[:, :3].astype(np.float32),
                vector_fields.astype(np.float32),
                wall_mask.astype(np.uint16),
                np.int32(w), np.int32(h), np.int32(d),
                vs, mode_sign
            )

        vertices[:, 3] = thicknesses
        return vertices

    def _save_plt(self, fname, vertices, vertices_list):
        """Save thickness results in Tecplot PLT format."""
        elem_cnt = len(vertices)
        if elem_cnt == 0:
            return

        vol_size = self.m_volume_size
        spacing = self.m_voxel_spacing
        vol_pos = self.m_volume_position

        print(f"m_volume_size = {vol_size[0]}, {vol_size[1]}, {vol_size[2]}")
        print(f"m_voxel_spacing = {spacing[0]}, {spacing[1]}, {spacing[2]}")
        print(f"m_volume_position = {vol_pos[0]}, {vol_pos[1]}, {vol_pos[2]}")

        lines = []
        lines.append('VARIABLES = "X", "Y", "Z", "Thickness(mm)"\n')
        lines.append(f'ZONE I={elem_cnt} , DATAPACKING=POINT\n')

        average_wt = 0.0

        for i in range(elem_cnt):
            vx = vertices[i, 0]
            vy = vertices[i, 1]
            vz = vertices[i, 2]
            vw = vertices[i, 3]

            vx /= vol_size[0]
            vy /= vol_size[1]
            vz /= vol_size[2]

            vx = vx * (spacing[0] * vol_size[0])
            vy = vy * (spacing[1] * vol_size[1])
            vz = (1.0 - vz) * (spacing[2] * vol_size[2])

            vx = vol_pos[0] - 0.0 + vx
            vy = vol_pos[1] - 0.0 + vy
            vz = vol_pos[2] - (vol_size[2] * spacing[2]) + vz

            lines.append(f"{vx} {vy} {vz} {vw}\n")
            average_wt += vw

            vertices_list.append([vx, vy, vz, vw])

        print(f"{fname}-average WT = {average_wt / elem_cnt}")

        with open(fname + ".plt", 'w') as f:
            f.writelines(lines)
