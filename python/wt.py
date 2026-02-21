"""Wall Thickness detection and computation.
Ported from WT.cpp/h and CUDA kernels in WT_kernel.cu/cuh
"""

import os
import sys
import math
import numpy as np
from scipy import ndimage

from backend import xp, to_numpy, to_device, GPU_AVAILABLE, get_scipy_ndimage
from utils import export_bmp_wt, normalize_float_to_uint16

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


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
    """Invert a uint16 mask: <1 -> 1, else -> 0.
    Port of inverseMask3D_kernel for uint16 from WT_kernel.cuh:101-118
    """
    return np.where(mask < 1, np.uint16(1), np.uint16(0))


def inverse_mask_float(mask):
    """Invert a float mask: <1 -> 1.0, else -> 0.0.
    Port of inverseMask3D_kernel for float from WT_kernel.cuh:101-118
    """
    return np.where(mask < 1.0, 1.0, 0.0).astype(np.float32)


def fillup_volume_by_mask(mask, output, set_value, base_value=0):
    """Fill output where mask == base_value with set_value.
    Port of fillupVolumebyMask_kernel from WT_kernel.cuh:73-86

    Args:
        mask: uint16 3D array
        output: float32 3D array (modified in-place)
        set_value: float value to set
        base_value: uint16 value to match in mask
    """
    output[mask == base_value] = set_value


def cutoff_volume(volume, cutoff, set_value):
    """Set voxels <= cutoff to set_value.
    Port of cutoffVolume_kernel from WT_kernel.cuh:120-133

    Args:
        volume: float32 3D array (modified in-place)
        cutoff: threshold
        set_value: replacement value
    """
    volume[volume <= cutoff] = set_value


def binarize(volume, cutoff):
    """Binarize volume: >cutoff -> 1.0, else -> 0.0
    Port of binalized_kernel from WT_kernel.cuh:136-152

    Args:
        volume: float32 3D array

    Returns:
        binarized float32 3D array
    """
    return np.where(volume > cutoff, 1.0, 0.0).astype(np.float32)


def subtract_by_bool(float_buf, bool_buf):
    """Zero out float_buf where bool_buf > 0.
    Port of subtract_by_bool_kernel from WT_kernel.cuh:220-233

    Args:
        float_buf: float32 3D array (modified in-place)
        bool_buf: uint16 3D array
    """
    float_buf[bool_buf > 0] = 0.0


def connectivity_filtering(wall_mask, condition, set_value):
    """For wall voxels with any 26-neighbor > 0 in condition, set output to set_value.
    Port of connectivityFiltering_kernel from WT_kernel.cuh:187-217

    Args:
        wall_mask: uint16 3D array (wall voxels)
        condition: float32 3D array (condition to check neighbors)
        set_value: float value to assign

    Returns:
        output: float32 3D array
    """
    d, h, w = wall_mask.shape
    output = np.zeros_like(condition, dtype=np.float32)

    # Check 26-neighborhood using maximum_filter
    # The kernel checks all 27 neighbors (including self) for condition > 0
    neighbor_max = ndimage.maximum_filter(condition, size=3, mode='constant', cval=0.0)

    # Where wall_mask > 0 AND any neighbor in condition > 0
    mask = (wall_mask > 0) & (neighbor_max > 0)
    output[mask] = set_value

    return output


def compute_laplace_equation(in_vol, wall_mask, iterations):
    """Solve Laplace equation with boundary conditions.
    Port of computeLaplaceEquation from WT_kernel.cu:112-144
    Uses GPU (CuPy) when available, CPU (SciPy) otherwise.

    Optimized: pre-allocated gradient arrays, reduced temporaries.

    Args:
        in_vol: float32 3D input volume
        wall_mask: uint16 3D mask (boundary condition: output *= mask)
        iterations: max number of iterations

    Returns:
        result: float32 3D array (the output buffer from last iteration)
    """
    ndi = get_scipy_ndimage()

    # 6-neighbor averaging kernel
    kernel = xp.zeros((3, 3, 3), dtype=xp.float32)
    kernel[1, 1, 0] = 1.0 / 6.0
    kernel[1, 1, 2] = 1.0 / 6.0
    kernel[1, 0, 1] = 1.0 / 6.0
    kernel[1, 2, 1] = 1.0 / 6.0
    kernel[0, 1, 1] = 1.0 / 6.0
    kernel[2, 1, 1] = 1.0 / 6.0

    in_ptr = to_device(in_vol.copy().astype(np.float32))
    out_ptr = xp.zeros_like(in_ptr)
    mask_float = to_device(wall_mask.astype(np.float32))

    # Pre-allocate gradient arrays (avoid re-allocation each iteration)
    dx = xp.zeros_like(in_ptr)
    dy = xp.zeros_like(in_ptr)
    dz = xp.zeros_like(in_ptr)

    h_epsilon = 0.0

    for iter_idx in range(iterations):
        p_ei = h_epsilon

        # Laplace step: average 6 neighbors
        ndi.convolve(in_ptr, kernel, output=out_ptr, mode='constant', cval=0.0)
        out_ptr *= mask_float

        # Compute gradient magnitude sum (epsilon)
        dx[:] = 0
        dy[:] = 0
        dz[:] = 0
        dx[:, :, 1:-1] = (in_ptr[:, :, 2:] - in_ptr[:, :, :-2]) * 0.5
        dy[:, 1:-1, :] = (in_ptr[:, 2:, :] - in_ptr[:, :-2, :]) * 0.5
        dz[1:-1, :, :] = (in_ptr[2:, :, :] - in_ptr[:-2, :, :]) * 0.5

        # grad_mag is always >= 0, so sum(mag[mag>0]) == sum(mag)
        h_epsilon = float(xp.sum(xp.sqrt(dx * dx + dy * dy + dz * dz)))

        p_ei_1 = h_epsilon

        # Swap buffers
        in_ptr, out_ptr = out_ptr, in_ptr

        if p_ei > 0:
            err = abs((p_ei - p_ei_1) / p_ei)
            if err < 1e-5 or iter_idx > iterations:
                print(f"e={err}", file=sys.stderr)
                break
            print(f"iter = {iter_idx}, E={p_ei}, next E={p_ei_1}, ERR= {err}", file=sys.stderr)

    return to_numpy(in_ptr)


def compute_laplace_with_vector(in_vol, G_field, wall_mask, iterations):
    """Solve Laplace equation and compute gradient vector field.
    Port of computeLaplaceEquation_with_Vector from WT_kernel.cu:146-179
    Uses GPU (CuPy) when available, CPU (SciPy) otherwise.

    Args:
        in_vol: float32 3D input volume
        G_field: float32 4D array (D, H, W, 4) - gradient + w component
        wall_mask: uint16 3D mask (not directly used in kernel, but for reference)
        iterations: max number of iterations

    Returns:
        result: float32 3D array (output volume)
        G_field: modified gradient field
    """
    ndi = get_scipy_ndimage()

    kernel = xp.zeros((3, 3, 3), dtype=xp.float32)
    kernel[1, 1, 0] = 1.0 / 6.0
    kernel[1, 1, 2] = 1.0 / 6.0
    kernel[1, 0, 1] = 1.0 / 6.0
    kernel[1, 2, 1] = 1.0 / 6.0
    kernel[0, 1, 1] = 1.0 / 6.0
    kernel[2, 1, 1] = 1.0 / 6.0

    in_ptr = to_device(in_vol.copy().astype(np.float32))
    out_ptr = xp.zeros_like(in_ptr)
    d_G_field = to_device(G_field)

    # Pre-allocate gradient arrays
    dx = xp.zeros_like(in_ptr)
    dy = xp.zeros_like(in_ptr)
    dz = xp.zeros_like(in_ptr)
    grad_mag = xp.zeros_like(in_ptr)

    h_epsilon = 0.0

    for iter_idx in range(iterations):
        p_ei = h_epsilon

        # Laplace step (no mask multiplication unlike the other version)
        ndi.convolve(in_ptr, kernel, output=out_ptr, mode='constant', cval=0.0)

        # Compute gradient (central differences)
        dx[:] = 0
        dy[:] = 0
        dz[:] = 0
        dx[:, :, 1:-1] = (in_ptr[:, :, 2:] - in_ptr[:, :, :-2]) * 0.5
        dy[:, 1:-1, :] = (in_ptr[:, 2:, :] - in_ptr[:, :-2, :]) * 0.5
        dz[1:-1, :, :] = (in_ptr[2:, :, :] - in_ptr[:-2, :, :]) * 0.5

        xp.sqrt(dx * dx + dy * dy + dz * dz, out=grad_mag)

        # Normalize gradient and store in G_field where magnitude > 0
        valid = grad_mag > 0
        if xp.any(valid):
            inv_mag = 1.0 / grad_mag[valid]
            d_G_field[valid, 0] = dx[valid] * inv_mag
            d_G_field[valid, 1] = dy[valid] * inv_mag
            d_G_field[valid, 2] = dz[valid] * inv_mag

        h_epsilon = float(xp.sum(grad_mag))
        p_ei_1 = h_epsilon

        # Swap
        in_ptr, out_ptr = out_ptr, in_ptr

        if p_ei > 0:
            err = abs((p_ei - p_ei_1) / p_ei)
            if err < 1e-5 or iter_idx > 400:
                print(f"e={err}", file=sys.stderr)
                break
            print(f"iter = {iter_idx}, E={p_ei}, next E={p_ei_1}, ERR= {err}", file=sys.stderr)

    return to_numpy(in_ptr), to_numpy(d_G_field)


def cuda_ccl(volume, degree_of_connectivity=4):
    """Connected Component Labeling - keep only the largest component.
    Port of cuda_ccl from WT_kernel.cu:191-252

    Uses scipy.ndimage.label instead of the iterative CUDA approach,
    but produces equivalent results.

    The C++ CCL uses 6-connectivity (Manhattan distance <= 1) when degree=4,
    then finds the largest component and keeps only it.
    Final step: memcpy_uint32_to_float with cond=false means
    largest component -> 0, rest -> 1. But actually looking at the code more carefully:
    remain_largest_CCL sets matching labels to 1, others to 0.
    Then memcpy_uint32_to_float with cond=false: >0 -> 0, else -> 1.
    So the FINAL result is: largest component = 0.0, everything else = 1.0

    Wait, let me re-read: memcpy_uint32_to_float with cond=false (line 249):
    if in_buffer[idx] > 0: out = 0.0, else: out = 1.0
    So largest CCL region = 1 in uint32 -> becomes 0.0 in float
    Background (0 in uint32) -> becomes 1.0 in float

    Args:
        volume: float32 3D array (values >= 1.0 are foreground)
        degree_of_connectivity: connectivity degree (4 = 6-connected in 3D)

    Returns:
        result: float32 3D array
    """
    # Create binary mask (foreground = values >= 1.0)
    binary = (volume >= 1.0).astype(np.int32)

    # 6-connected structure (Manhattan distance <= 1)
    struct = ndimage.generate_binary_structure(3, 1)

    # Label connected components
    labeled, num_features = ndimage.label(binary, structure=struct)

    if num_features == 0:
        # No components - return inverted (all 1.0)
        return np.ones_like(volume, dtype=np.float32)

    # Find the largest component (exclude background label 0)
    component_sizes = np.bincount(labeled.ravel())
    # Skip label 0 (background)
    if len(component_sizes) > 1:
        component_sizes[0] = 0
        largest_label = np.argmax(component_sizes)
    else:
        return np.ones_like(volume, dtype=np.float32)

    print(f"CCL1 min/max:0, {int(np.max(component_sizes))}")
    print(f"CCL argmax:{largest_label}")

    # Keep only the largest component
    # remain_largest_CCL: matching -> 1, else -> 0
    result_uint = np.where(labeled == largest_label, 1, 0).astype(np.uint32)

    # memcpy_uint32_to_float with cond=false: >0 -> 0.0, else -> 1.0
    result = np.where(result_uint > 0, 0.0, 1.0).astype(np.float32)

    return result


# ============================================================
# Numba JIT-accelerated thickness computation
# ============================================================

def _compute_thickness_single(start_x, start_y, start_z,
                               vector_fields, wall_mask,
                               w, h, d, voxel_size, mode_sign):
    """Compute thickness for a single vertex. Pure Python (no numpy calls)."""
    MAX_TRAVEL = 256
    dt = 0.001
    max_steps = int(MAX_TRAVEL / dt)

    sx = int(start_x)
    sy = int(start_y)
    sz = int(start_z)

    if sx >= w or sy >= h or sz >= d or sx < 0 or sy < 0 or sz < 0:
        return 0.0

    # Integration path (start at voxel center)
    px = start_x + 0.5
    py = start_y + 0.5
    pz = start_z + 0.5

    dx = 0.0
    dy = 0.0
    dz = 0.0
    state = True
    thickness = 0.0

    ix = sx
    iy = sy
    iz = sz

    for step in range(max_steps):
        if ix >= w or iy >= h or iz >= d or ix < 0 or iy < 0 or iz < 0:
            break

        vf0 = vector_fields[iz, iy, ix, 0]
        vf1 = vector_fields[iz, iy, ix, 1]
        vf2 = vector_fields[iz, iy, ix, 2]
        vf3 = vector_fields[iz, iy, ix, 3]

        if vf3 > 0:
            state = False
            dot_val = dx * vf0 + dy * vf1 + dz * vf2

            if dot_val >= 0:
                dx = vf0 * mode_sign
                dy = vf1 * mode_sign
                dz = vf2 * mode_sign
                dir_len = math.sqrt(dx * dx + dy * dy + dz * dz)

                if dir_len > 0:
                    prev_px = px
                    prev_py = py
                    prev_pz = pz

                    px += dx * dt
                    py += dy * dt
                    pz += dz * dt

                    ix = int(px)
                    iy = int(py)
                    iz = int(pz)

                    if ix >= w or iy >= h or iz >= d or ix < 0 or iy < 0 or iz < 0:
                        break

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
                else:
                    break
            else:
                if step == 0:
                    dx = vf0 * mode_sign
                    dy = vf1 * mode_sign
                    dz = vf2 * mode_sign
                    px += dx * dt
                    py += dy * dt
                    pz += dz * dt
                    ix = int(px)
                    iy = int(py)
                    iz = int(pz)
                    if ix >= w or iy >= h or iz >= d or ix < 0 or iy < 0 or iz < 0:
                        break
                    continue
                break
        else:
            if not state:
                break
            else:
                dx = vf0 * mode_sign
                dy = vf1 * mode_sign
                dz = vf2 * mode_sign
                px += dx * dt
                py += dy * dt
                pz += dz * dt
                ix = int(px)
                iy = int(py)
                iz = int(pz)
                if ix >= w or iy >= h or iz >= d or ix < 0 or iy < 0 or iz < 0:
                    break

    return thickness


if HAS_NUMBA:
    # Numba JIT-compiled version (near-C performance)
    _thickness_single_jit = njit(_compute_thickness_single, cache=True)

    @njit(parallel=True, cache=True)
    def _compute_thickness_batch(vertices_xyz, vector_fields, wall_mask,
                                  w, h, d, voxel_size, mode_sign):
        """Batch compute thickness for all vertices using Numba parallel."""
        N = vertices_xyz.shape[0]
        result = np.zeros(N, dtype=np.float32)

        for idx in prange(N):
            result[idx] = _thickness_single_jit(
                vertices_xyz[idx, 0], vertices_xyz[idx, 1], vertices_xyz[idx, 2],
                vector_fields, wall_mask,
                w, h, d, voxel_size, mode_sign
            )
            if idx % 50000 == 0 and idx > 0:
                print("  thickness progress:", idx, "/", N)

        return result
else:
    # Fallback: pure Python (slower but works everywhere)
    def _compute_thickness_batch(vertices_xyz, vector_fields, wall_mask,
                                  w, h, d, voxel_size, mode_sign):
        """Batch compute thickness - pure Python fallback."""
        N = vertices_xyz.shape[0]
        result = np.zeros(N, dtype=np.float32)

        for idx in range(N):
            result[idx] = _compute_thickness_single(
                vertices_xyz[idx, 0], vertices_xyz[idx, 1], vertices_xyz[idx, 2],
                vector_fields, wall_mask,
                w, h, d, voxel_size, mode_sign
            )
            if idx % 1000 == 0 and idx > 0:
                print(f"  thickness progress: {idx}/{N}", file=sys.stderr)

        return result


# ============================================================
# Main WT class
# ============================================================

class WT:
    """Wall Thickness computation class.
    Exact port of WT class from WT.cpp/h
    """

    def __init__(self, save_path, volume_size, voxel_spacing, volume_position, wall_mask, convex_mask):
        """
        Args:
            save_path: output directory path
            volume_size: (width, height, depth) or (w, h, d, c) tuple
            voxel_spacing: numpy array [x, y, z]
            volume_position: numpy array [x, y, z]
            wall_mask: uint16 3D array (depth, height, width)
            convex_mask: uint16 3D array (depth, height, width)
        """
        self.m_save_path = save_path

        if len(volume_size) == 4:
            self.m_volume_size = np.array([volume_size[0], volume_size[1], volume_size[2]], dtype=np.float32)
        else:
            self.m_volume_size = np.array(volume_size[:3], dtype=np.float32)

        self.m_voxel_spacing = voxel_spacing.copy().astype(np.float32)
        self.m_volume_position = volume_position.copy().astype(np.float32)

        # Copy masks
        self.m_wall_mask = wall_mask.copy().astype(np.uint16)
        self.m_convex_mask = convex_mask.copy().astype(np.uint16)
        self.m_chamber_mask = np.zeros_like(wall_mask, dtype=np.uint16)

        # Will be set during computation
        self.m_VFInit = None
        self.m_vectorfields = None

        # Output
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
        mem_size = w * h * d

        d_vf3D_frontbuf = np.zeros((d, h, w), dtype=np.float32)
        d_vf3D_backbuf = np.zeros((d, h, w), dtype=np.float32)

        # Invert convex mask and wall mask: inner=0, outer=1 -> inner=1
        self.m_convex_mask = inverse_mask_uint16(self.m_convex_mask)
        self.m_wall_mask = inverse_mask_uint16(self.m_wall_mask)

        # Initialize: fill convex boundary with 10.0
        fillup_volume_by_mask(self.m_convex_mask, d_vf3D_frontbuf, 10.0, base_value=1)

        # Laplace equation (100 iterations)
        d_vf3D_backbuf = compute_laplace_equation(
            d_vf3D_frontbuf, self.m_wall_mask, 100
        )

        # Cutoff: values <= 1e-01 -> -2.0
        cutoff_volume(d_vf3D_backbuf, 1e-01, -2.0)

        # Binarize: > -2.0 -> 1.0, else -> 0.0
        d_vf3D_backbuf = binarize(d_vf3D_backbuf, -2.0)

        # Invert mask (exterior -> interior)
        d_vf3D_backbuf = inverse_mask_float(d_vf3D_backbuf)

        # Restore wall mask
        self.m_wall_mask = inverse_mask_uint16(self.m_wall_mask)

        # Subtract wall from chamber region
        subtract_by_bool(d_vf3D_backbuf, self.m_wall_mask)

        # Free convex mask (not needed anymore)
        self.m_convex_mask = None

        # CCL: keep largest connected component
        d_vf3D_backbuf = cuda_ccl(d_vf3D_backbuf, 4)

        # Subtract wall again
        subtract_by_bool(d_vf3D_backbuf, self.m_wall_mask)

        # Normalize and store chamber mask
        norm_buf = normalize_float_to_uint16(d_vf3D_backbuf)
        self.m_chamber_mask = norm_buf.astype(np.uint16)

        # Store VFInit (inner/wall/outer initialization)
        self.m_VFInit = d_vf3D_backbuf.copy()

        # Fill wall region in VFInit with 0.5
        fillup_volume_by_mask(self.m_wall_mask, self.m_VFInit, 0.5, base_value=1)

        # Initialize frontbuf with wall = 2.0
        d_vf3D_frontbuf = np.zeros((d, h, w), dtype=np.float32)
        fillup_volume_by_mask(self.m_wall_mask, d_vf3D_frontbuf, 2.0, base_value=1)

        # Epicardium detection: connectivity filtering with setValue=3.0
        epi_result = connectivity_filtering(self.m_wall_mask, d_vf3D_backbuf, 3.0)
        # Merge into frontbuf
        d_vf3D_frontbuf = np.where(epi_result > 0, epi_result, d_vf3D_frontbuf)

        # Endocardium detection
        d_vf3D_backbuf = inverse_mask_float(d_vf3D_backbuf)
        subtract_by_bool(d_vf3D_backbuf, self.m_wall_mask)
        endo_result = connectivity_filtering(self.m_wall_mask, d_vf3D_backbuf, 1.0)
        # Merge into frontbuf
        d_vf3D_frontbuf = np.where(endo_result > 0, endo_result, d_vf3D_frontbuf)

        # Store vector fields (zone labels: 1=endo, 2=wall, 3=epi)
        self.m_vectorfields = d_vf3D_frontbuf.copy()

        # Export BMP of Epi-Endo result
        export_bmp_wt(d_vf3D_frontbuf, (w, h, d), self.m_save_path, "Epi-Endo")

    def eval_wt(self):
        """Evaluate wall thickness using Laplace equation + Euler's method.
        Exact port of WT::evalWT() from WT.cpp:313-415
        """
        w = int(self.m_volume_size[0])
        h = int(self.m_volume_size[1])
        d = int(self.m_volume_size[2])
        mem_size = w * h * d

        # Initialize voltage from VFInit
        voltage3D = self.m_VFInit.copy()
        voltage3D_backup = np.zeros((d, h, w), dtype=np.float32)

        # Vector fields (4 components: xyz gradient + w)
        vector_fields = np.zeros((d, h, w, 4), dtype=np.float32)
        vector_fields[:, :, :, 3] = 1.0  # w = 1.0

        # Laplace equation with vector field (400 iterations)
        voltage3D_backup, vector_fields = compute_laplace_with_vector(
            voltage3D, vector_fields, self.m_wall_mask, 400
        )

        # Extract endo and epi vertices (vectorized)
        h_vectorfields = self.m_vectorfields

        endo_coords = np.argwhere(h_vectorfields == 1.0)  # (N, 3) = (z, y, x)
        if len(endo_coords) > 0:
            endo_vertices = np.zeros((len(endo_coords), 4), dtype=np.float32)
            endo_vertices[:, 0] = endo_coords[:, 2].astype(np.float32)  # x
            endo_vertices[:, 1] = endo_coords[:, 1].astype(np.float32)  # y
            endo_vertices[:, 2] = endo_coords[:, 0].astype(np.float32)  # z
        else:
            endo_vertices = np.zeros((0, 4), dtype=np.float32)

        epi_coords = np.argwhere(h_vectorfields == 3.0)
        if len(epi_coords) > 0:
            epi_vertices = np.zeros((len(epi_coords), 4), dtype=np.float32)
            epi_vertices[:, 0] = epi_coords[:, 2].astype(np.float32)
            epi_vertices[:, 1] = epi_coords[:, 1].astype(np.float32)
            epi_vertices[:, 2] = epi_coords[:, 0].astype(np.float32)
        else:
            epi_vertices = np.zeros((0, 4), dtype=np.float32)

        endo_cnt = len(endo_vertices)
        epi_cnt = len(epi_vertices)
        print(f"total_endoVCnt_size = {endo_cnt}, {epi_cnt}", file=sys.stderr)

        # Compute thickness for endo vertices
        voxel_size = self.m_voxel_spacing.copy()
        endo_vertices = self._compute_thickness(
            endo_vertices, vector_fields, self.m_wall_mask, (w, h, d), voxel_size, mode=0
        )

        # Save PLT
        self.m_endo_vertices_list = []
        self._save_plt(
            os.path.join(self.m_save_path, "WT-endo"),
            endo_vertices, self.m_endo_vertices_list
        )

    def _compute_thickness(self, vertices, vector_fields, wall_mask, vol_size, voxel_size, mode=0):
        """Compute wall thickness for each vertex using Euler's method.
        Uses Numba JIT for near-C performance.
        Exact port of compute_thickness_kernel from WT_kernel.cuh:485-573
        """
        w, h, d = vol_size
        mode_sign = np.float32(-1.0 if mode else 1.0)
        vs = voxel_size.astype(np.float32)

        # Call Numba-accelerated batch function
        thicknesses = _compute_thickness_batch(
            vertices[:, :3].astype(np.float32),
            vector_fields.astype(np.float32),
            wall_mask.astype(np.uint16),
            np.int32(w), np.int32(h), np.int32(d),
            vs, mode_sign
        )
        vertices[:, 3] = thicknesses
        return vertices

    def _save_plt(self, fname, vertices, vertices_list):
        """Save thickness results in Tecplot PLT format.
        Exact port of WT::savePLT() from WT.cpp:417-462

        Args:
            fname: output filename (without .plt extension)
            vertices: Nx4 float array (x, y, z, thickness)
            vertices_list: list to append transformed vertices to
        """
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

            # Normalize (0 ~ 1)
            vx /= vol_size[0]
            vy /= vol_size[1]
            vz /= vol_size[2]

            # Scale to physical coordinates
            vx = vx * (spacing[0] * vol_size[0])
            vy = vy * (spacing[1] * vol_size[1])
            vz = (1.0 - vz) * (spacing[2] * vol_size[2])

            # Offset by volume position
            vx = vol_pos[0] - 0.0 + vx
            vy = vol_pos[1] - 0.0 + vy
            vz = vol_pos[2] - (vol_size[2] * spacing[2]) + vz

            lines.append(f"{vx} {vy} {vz} {vw}\n")
            average_wt += vw

            vertices_list.append([vx, vy, vz, vw])

        print(f"{fname}-average WT = {average_wt / elem_cnt}")

        with open(fname + ".plt", 'w') as f:
            f.writelines(lines)
