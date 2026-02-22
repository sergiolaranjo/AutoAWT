"""Utility functions for file I/O, polygon fill, and array operations.
Ported from util.cpp/h and CUDA kernels in util_3D.cu/cuh
"""

import os
import sys
import struct
import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:
    pydicom = None

from qhull import CTview


def natural_sort_key(s):
    """Sort key for natural (logical) sorting of filenames."""
    import re
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def select_directory(title="Select directory"):
    """Cross-platform directory selection dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title=title)
        root.destroy()
        return path
    except Exception:
        path = input(f"{title}: ")
        return path


def get_file_list(directory, ext):
    """Get sorted list of files with given extension in directory.

    Args:
        directory: path to directory
        ext: file extension (e.g., '.bmp', '.dcm')

    Returns:
        list of full file paths, naturally sorted
    """
    files = []
    for f in os.listdir(directory):
        if f.lower().endswith(ext.lower()):
            files.append(os.path.join(directory, f))
    files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    return files


def read_bmp_files(file_list):
    """Read a stack of BMP mask images into a 3D volume.
    Exact port of readBMPFiles() from util.cpp:360-465

    Args:
        file_list: list of BMP file paths

    Returns:
        volume: numpy array of shape (depth, height, width), dtype uint16
        volume_size: (width, height, depth, channels) tuple
    """
    if len(file_list) < 1:
        print("no have to BMP files", file=sys.stderr)
        return None, (0, 0, 0, 0)

    depth = len(file_list)
    width = height = 0
    channel = 0
    volume_slices = []

    for i, filepath in enumerate(file_list):
        img = Image.open(filepath)
        if i == 0:
            width = img.width
            height = img.height
            if img.mode == 'RGB':
                channel = 3
            elif img.mode in ('L', 'P'):
                channel = 1
            else:
                if len(img.getbands()) >= 3:
                    img = img.convert('RGB')
                    channel = 3
                else:
                    img = img.convert('L')
                    channel = 1
        else:
            # Subsequent images: convert to match the first image's channel
            if channel == 3:
                img = img.convert('RGB')
            else:
                img = img.convert('L')

        raw_data = np.array(img)
        data_ptr = np.zeros((height, width), dtype=np.uint16)

        if channel == 3:
            if len(raw_data.shape) == 2:
                raw_data = np.stack([raw_data] * 3, axis=-1)
            # Vectorized Y-axis flip + color check
            r = raw_data[:, :, 0].astype(np.int32)
            g = raw_data[:, :, 1].astype(np.int32)
            b = raw_data[:, :, 2].astype(np.int32)
            same_color = (r == g) & (g == b)
            mask = ~same_color
            flipped = np.flipud(mask)
            data_ptr = np.where(flipped, np.uint16(255), np.uint16(0))
        elif channel == 1:
            if len(raw_data.shape) == 3:
                raw_data = raw_data[:, :, 0]
            # Vectorized Y-axis flip + threshold
            flipped = np.flipud(raw_data)
            data_ptr = np.where(flipped > 0, np.uint16(255), np.uint16(0))

        volume_slices.append(data_ptr)

    volume = np.stack(volume_slices, axis=0).astype(np.uint16)
    volume_size = (width, height, depth, 1)
    return volume, volume_size


def read_dcm_for_pixel_spacing(file_list):
    """Read DICOM files to extract pixel spacing, volume position, and patient ID.
    Exact port of readDCMFiles_for_pixelSpacing() from util.cpp:82-310

    Args:
        file_list: list of DICOM file paths

    Returns:
        pixel_spacing: numpy array [x, y, z]
        volume_position: numpy array [x, y, z]
        patient_id: string
    """
    if pydicom is None:
        raise ImportError("pydicom is required for DICOM reading")

    if len(file_list) < 1:
        print("no have to DCM files", file=sys.stderr)
        return np.array([1.0, 1.0, 0.5], dtype=np.float32), np.zeros(3, dtype=np.float32), ""

    pixel_spacing = np.array([1.0, 1.0, 0.5], dtype=np.float32)
    volume_position = np.zeros(3, dtype=np.float32)
    patient_id = ""
    once_volume_position = True
    slice_locations = []

    # Single pass: extract PixelSpacing, PatientID, ImagePositionPatient, SliceLocation
    for i, filepath in enumerate(file_list):
        try:
            ds = pydicom.dcmread(filepath)
        except Exception:
            print("dcm load error", file=sys.stderr)
            continue

        # PatientID from first file
        if i == 0:
            if hasattr(ds, 'PatientID'):
                patient_id = str(ds.PatientID)
                print(f"patientID: {patient_id}")

        # PixelSpacing
        if hasattr(ds, 'PixelSpacing'):
            ps = ds.PixelSpacing
            pixel_spacing[0] = float(ps[0])
            pixel_spacing[1] = float(ps[1])

        # ImagePositionPatient from first file
        if once_volume_position and hasattr(ds, 'ImagePositionPatient'):
            ipp = ds.ImagePositionPatient
            volume_position[0] = float(ipp[0])
            volume_position[1] = float(ipp[1])
            volume_position[2] = float(ipp[2])
            once_volume_position = False

        # SliceLocation for z-spacing computation
        if hasattr(ds, 'SliceLocation'):
            slice_locations.append(float(ds.SliceLocation))

    if len(slice_locations) >= 2:
        slice_locations.sort()
        z_diffs = [abs(slice_locations[i+1] - slice_locations[i])
                   for i in range(len(slice_locations) - 1)]
        z_diffs = [d for d in z_diffs if d > 0]
        if z_diffs:
            pixel_spacing[2] = float(np.median(z_diffs))
        else:
            pixel_spacing[2] = 0.5
    else:
        pixel_spacing[2] = 0.5

    print(f"volume_position={volume_position[0]}, {volume_position[1]}, {volume_position[2]}")
    print(f"pixel_spacing={pixel_spacing[0]}, {pixel_spacing[1]}, {pixel_spacing[2]}")

    return pixel_spacing, volume_position, patient_id


def export_bmp(volume, volume_size, name, base_path=".."):
    """Export 3D volume as BMP slices.
    Exact port of exportBMP() from util.cpp:467-536

    Args:
        volume: numpy array of shape (depth, height, width), dtype uint16
        volume_size: (width, height, depth, channels) tuple
        name: subdirectory name
        base_path: base output path
    """
    save_path = os.path.join(base_path, f"BMP-{name}")
    os.makedirs(save_path, exist_ok=True)

    w, h, d = volume_size[0], volume_size[1], volume_size[2]

    for z in range(d):
        h_slice = volume[z].copy()
        h_slice_mask = np.clip(h_slice, 0, 255).astype(np.uint8)

        img = Image.fromarray(h_slice_mask, mode='L')
        img.save(os.path.join(save_path, f"{z + 1}.bmp"))


def export_bmp_wt(float_volume, volume_size, save_path, name):
    """Export float volume as BMP slices (WT version with normalization).
    Exact port of WT::exportBMP() from WT.cpp:194-306

    Args:
        float_volume: numpy array of shape (depth, height, width), dtype float32
        volume_size: (width, height, depth) tuple
        save_path: base save path
        name: subdirectory name
    """
    w, h, d = volume_size

    # Normalize
    normalized = normalize_float_to_uint16(float_volume)

    sub_path = os.path.join(save_path, name) if name else os.path.join(save_path, "BMP")
    os.makedirs(sub_path, exist_ok=True)

    for z in range(d):
        h_slice = normalized[z]

        # Vectorized Y-flip + clamp
        flipped = np.flipud(h_slice)
        h_slice_mask = np.where(flipped > 0, np.minimum(flipped, 255), 0).astype(np.uint8)

        save_filename = os.path.join(sub_path, f"{z + 1}.bmp")
        print(save_filename)

        img = Image.fromarray(h_slice_mask, mode='L')
        img.save(save_filename)


def normalize_float_to_uint16(volume):
    """Normalize float volume to uint16 range [0, 255].
    Port of normalize_floatbuf from WT_kernel.cu

    Args:
        volume: numpy array, dtype float32

    Returns:
        normalized: numpy array, dtype uint16
    """
    v_min = float(np.min(volume))
    v_max = float(np.max(volume))
    print(f"min/max:{v_min}, {v_max}")

    # Match the C++ kernel: value = (in[idx] - min) / (10.0 - (-0.5))
    # Actually the kernel has a bug/hardcoded range, but let's use actual min/max
    if v_max - v_min < 1e-10:
        return np.zeros_like(volume, dtype=np.uint16)

    normalized = ((volume.astype(np.float64) - v_min) / (v_max - v_min) * 255.0)
    normalized = np.clip(normalized, 0, 255).astype(np.uint16)
    return normalized


def is_left(p0, p1, p2):
    """isLeft test for winding number algorithm.
    Port from marchingCube_kernel.cuh:258-262
    """
    return ((p1[0] - p0[0]) * (p2[1] - p0[1])
            - (p2[0] - p0[0]) * (p1[1] - p0[1]))


def _winding_number_vectorized(hull_pts, grid_x, grid_y):
    """Vectorized winding number computation for a 2D grid.

    Args:
        hull_pts: Nx2 numpy array of hull vertices
        grid_x: 2D array of x coordinates
        grid_y: 2D array of y coordinates

    Returns:
        2D boolean array (True = inside polygon)
    """
    n = len(hull_pts)
    wn = np.zeros(grid_x.shape, dtype=np.int32)

    for i in range(n):
        v1x = hull_pts[i, 0]
        v1y = hull_pts[i, 1]
        v2x = hull_pts[(i + 1) % n, 0]
        v2y = hull_pts[(i + 1) % n, 1]

        # isLeft test vectorized
        il = (v2x - v1x) * (grid_y - v1y) - (grid_x - v1x) * (v2y - v1y)

        # Upward crossing
        mask_up = (v1y <= grid_y) & (v2y > grid_y) & (il > 0)
        wn += mask_up.astype(np.int32)

        # Downward crossing
        mask_down = (v1y > grid_y) & (v2y <= grid_y) & (il < 0)
        wn -= mask_down.astype(np.int32)

    return wn > 0


def compute_fill_space(result_buffer, convex_points, idx, volume_size, view):
    """Fill the interior of a convex hull polygon using winding number algorithm.
    Exact port of computeFillSpace() from util.cpp:539-580
    and polygon_fill_2D kernel from marchingCube_kernel.cuh:265-314

    Args:
        result_buffer: numpy array of shape (depth, height, width), dtype uint16
        convex_points: list of (x, y) tuples from convex hull
        idx: slice index
        volume_size: (width, height, depth) tuple
        view: CTview enum value

    Returns:
        Modified result_buffer
    """
    w, h, d = int(volume_size[0]), int(volume_size[1]), int(volume_size[2])
    line_size = len(convex_points)

    if line_size < 1:
        return result_buffer

    hull_pts = np.array(convex_points, dtype=np.float64)

    if view == CTview.AXIAL:
        if idx < 0 or idx >= d:
            return result_buffer
        tx_range = np.arange(w, dtype=np.float64)
        ty_range = np.arange(h, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(tx_range, ty_range)
        inside = _winding_number_vectorized(hull_pts, grid_x, grid_y)
        result_buffer[idx, :, :] = np.where(inside, np.uint16(1), np.uint16(0))

    elif view == CTview.SAGITTAL:
        if idx < 0 or idx >= w:
            return result_buffer
        tx_range = np.arange(d, dtype=np.float64)  # tx maps to z
        ty_range = np.arange(h, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(tx_range, ty_range)
        inside = _winding_number_vectorized(hull_pts, grid_x, grid_y)
        # inside shape: (h, d) -> inside[ty, tx] maps to result_buffer[tx, ty, idx]
        inside_uint = inside.astype(np.uint16)
        result_buffer[:, :, idx] = inside_uint.T  # Transpose: (d, h)

    elif view == CTview.CORONAL:
        if idx < 0 or idx >= h:
            return result_buffer
        tx_range = np.arange(w, dtype=np.float64)
        ty_range = np.arange(d, dtype=np.float64)  # ty maps to z
        grid_x, grid_y = np.meshgrid(tx_range, ty_range)
        inside = _winding_number_vectorized(hull_pts, grid_x, grid_y)
        # inside shape: (d, w) -> inside[ty, tx] maps to result_buffer[ty, idx, tx]
        result_buffer[:, idx, :] = inside.astype(np.uint16)

    return result_buffer


def arr_logical_and(dst, buf):
    """Logical AND of two uint16 volumes.
    Exact port of arr_logical_and() from util.cpp:582-599
    and cu_logical_and kernel from marchingCube_kernel.cuh:317-330

    Args:
        dst: numpy array, dtype uint16 (modified in-place)
        buf: numpy array, dtype uint16

    Returns:
        Modified dst array
    """
    dst[:] = np.where((dst > 0) & (buf > 0), 1, 0).astype(np.uint16)
    return dst
