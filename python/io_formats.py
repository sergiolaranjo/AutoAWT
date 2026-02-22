"""AutoAWT I/O Format Support

Multi-format volume import and export for medical imaging data.

Supported INPUT formats:
    - DICOM directory (*.dcm) — read pixel data directly, threshold to mask
    - BMP stack (*.bmp) — existing pipeline (binary segmentation masks)
    - NIfTI (*.nii, *.nii.gz) — requires nibabel
    - NRRD (*.nrrd) — requires pynrrd

Supported OUTPUT formats:
    - VTP (VTK PolyData XML) — for ParaView interactive 3D visualization
    - PLT (Tecplot) — existing format
    - STL (ASCII) — existing format

Usage:
    from io_formats import load_volume, export_vtp

    # Auto-detect format from path
    volume, spacing, position, patient_id = load_volume('/path/to/data')

    # Export mesh with thickness for ParaView
    export_vtp('output.vtp', vertices, faces, thickness)
"""

import os
import sys
import struct
import base64
import numpy as np


# ============================================================
# Volume Loading — Multi-format
# ============================================================

def load_volume(path, fmt=None, threshold=None, **kwargs):
    """Load a 3D volume from various medical imaging formats.

    Args:
        path: File or directory path
        fmt: Force format ('dicom', 'bmp', 'nifti', 'nrrd') or None for auto-detect
        threshold: For DICOM/NIfTI/NRRD, HU threshold for binary mask.
                   If None, uses format-specific default.
        **kwargs: Format-specific options:
            - dicom_window: (center, width) for windowing before threshold
            - mask_value: output value for mask voxels (default 255)

    Returns:
        (volume, volume_size, pixel_spacing, volume_position, patient_id)
        - volume: uint16 3D array (D, H, W) with 0/255 values
        - volume_size: (width, height, depth, channels)
        - pixel_spacing: [sx, sy, sz] in mm
        - volume_position: [x0, y0, z0] in mm
        - patient_id: string
    """
    if fmt is None:
        fmt = _detect_format(path)

    if fmt == 'bmp':
        return _load_bmp(path)
    elif fmt == 'dicom':
        return _load_dicom(path, threshold=threshold, **kwargs)
    elif fmt == 'nifti':
        return _load_nifti(path, threshold=threshold, **kwargs)
    elif fmt == 'nrrd':
        return _load_nrrd(path, threshold=threshold, **kwargs)
    else:
        raise ValueError(f"Unknown format: {fmt}. Use 'dicom', 'bmp', 'nifti', or 'nrrd'.")


def _detect_format(path):
    """Auto-detect format from path."""
    if os.path.isdir(path):
        files = os.listdir(path)
        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
        bmp_files = [f for f in files if f.lower().endswith('.bmp')]
        if bmp_files and not dcm_files:
            return 'bmp'
        if dcm_files and not bmp_files:
            return 'dicom'
        # Both present — check if enough DICOMs for a volume
        if len(dcm_files) > 10:
            return 'dicom'
        return 'bmp'
    else:
        ext = path.lower()
        if ext.endswith('.nii') or ext.endswith('.nii.gz'):
            return 'nifti'
        elif ext.endswith('.nrrd'):
            return 'nrrd'
        elif ext.endswith('.dcm'):
            return 'dicom'
        else:
            raise ValueError(f"Cannot detect format for: {path}")


# ============================================================
# DICOM Volume Import
# ============================================================

def _load_dicom(path, threshold=None, **kwargs):
    """Load DICOM series as 3D volume with optional thresholding.

    If the directory contains enough slices (>2), reads pixel data directly.
    Applies HU threshold to create binary mask.

    Args:
        path: Directory containing .dcm files, or single .dcm file
        threshold: HU threshold. Voxels > threshold become mask (255).
                   If None, uses Otsu's method or a default.
    """
    import pydicom

    if os.path.isfile(path):
        path = os.path.dirname(path)

    # Collect all DICOM files
    dcm_files = []
    for f in os.listdir(path):
        if f.lower().endswith('.dcm'):
            dcm_files.append(os.path.join(path, f))

    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {path}")

    # Read all slices and sort by SliceLocation
    slices = []
    for fp in dcm_files:
        try:
            ds = pydicom.dcmread(fp)
            if hasattr(ds, 'pixel_array'):
                slices.append(ds)
        except Exception as e:
            print(f"  Skipping {fp}: {e}", file=sys.stderr)

    if len(slices) < 2:
        raise ValueError(
            f"Only {len(slices)} DICOM slices with pixel data found. "
            "Need at least 2 slices for a volume. "
            "If your DICOMs are metadata-only, use --masks with BMP files."
        )

    # Sort by SliceLocation or InstanceNumber
    def sort_key(ds):
        if hasattr(ds, 'SliceLocation'):
            return float(ds.SliceLocation)
        if hasattr(ds, 'InstanceNumber'):
            return int(ds.InstanceNumber)
        return 0
    slices.sort(key=sort_key)

    # Extract metadata from first slice
    ds0 = slices[0]
    patient_id = str(getattr(ds0, 'PatientID', 'Unknown'))

    # Pixel spacing
    pixel_spacing = np.array([1.0, 1.0, 0.5], dtype=np.float32)
    if hasattr(ds0, 'PixelSpacing'):
        ps = ds0.PixelSpacing
        pixel_spacing[0] = float(ps[1])  # column spacing = X
        pixel_spacing[1] = float(ps[0])  # row spacing = Y

    # Z spacing from SliceLocation differences
    if len(slices) >= 2 and hasattr(slices[0], 'SliceLocation'):
        z_positions = [float(s.SliceLocation) for s in slices]
        z_diffs = [abs(z_positions[i+1] - z_positions[i]) for i in range(len(z_positions)-1)]
        z_diffs = [d for d in z_diffs if d > 0]
        if z_diffs:
            pixel_spacing[2] = min(z_diffs)

    # Volume position
    volume_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if hasattr(ds0, 'ImagePositionPatient'):
        ipp = ds0.ImagePositionPatient
        volume_position[0] = float(ipp[0])
        volume_position[1] = float(ipp[1])
        volume_position[2] = float(ipp[2])

    # Stack pixel arrays into 3D volume
    rows, cols = int(ds0.Rows), int(ds0.Columns)
    depth = len(slices)

    volume_raw = np.zeros((depth, rows, cols), dtype=np.float32)
    for i, ds in enumerate(slices):
        arr = ds.pixel_array.astype(np.float32)
        # Apply rescale slope/intercept for HU values
        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        arr = arr * slope + intercept
        # Flip Y axis (DICOM convention: row 0 is top)
        arr = arr[::-1, :]
        volume_raw[i] = arr

    print(f"DICOM volume: {cols}x{rows}x{depth}, "
          f"HU range: [{volume_raw.min():.0f}, {volume_raw.max():.0f}]",
          file=sys.stderr)

    # Threshold to binary mask
    if threshold is None:
        threshold = _otsu_threshold(volume_raw)
        print(f"Auto-threshold (Otsu): {threshold:.0f} HU", file=sys.stderr)

    mask_value = kwargs.get('mask_value', 255)
    volume = np.zeros_like(volume_raw, dtype=np.uint16)
    volume[volume_raw > threshold] = mask_value

    n_mask = np.count_nonzero(volume)
    n_total = volume.size
    print(f"Mask: {n_mask} voxels ({100*n_mask/n_total:.1f}% of volume)",
          file=sys.stderr)

    volume_size = (cols, rows, depth, 1)
    return volume, volume_size, pixel_spacing, volume_position, patient_id


def _otsu_threshold(data):
    """Compute Otsu's threshold for a volume."""
    # Subsample for speed
    flat = data.ravel()
    if len(flat) > 1_000_000:
        flat = flat[::len(flat)//1_000_000]

    # Remove background (very low values)
    flat = flat[flat > flat.min() + 1]

    # Histogram
    n_bins = 256
    hist, bin_edges = np.histogram(flat, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Otsu's method
    total = hist.sum()
    sum_total = (hist * bin_centers).sum()

    best_thresh = bin_centers[0]
    best_var = 0

    sum_bg = 0.0
    weight_bg = 0.0

    for i in range(n_bins):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += hist[i] * bin_centers[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = bin_centers[i]

    return best_thresh


# ============================================================
# BMP Stack Import (delegates to existing utils.py)
# ============================================================

def _load_bmp(path):
    """Load BMP stack using existing utils.read_bmp_files pipeline."""
    from utils import get_file_list, read_bmp_files, read_dcm_for_pixel_spacing

    file_names = get_file_list(path, '.bmp')
    if not file_names:
        raise FileNotFoundError(f"No BMP files found in {path}")

    volume, volume_size = read_bmp_files(file_names)
    if volume is None:
        raise RuntimeError("Failed to read BMP files")

    # Look for DICOM files in parent or sibling directories for metadata
    pixel_spacing = np.array([1.0, 1.0, 0.5], dtype=np.float32)
    volume_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    patient_id = "Unknown"

    # Check common sibling directories for DICOMs
    parent = os.path.dirname(path)
    for candidate in ['ct', 'dicom', 'dcm', 'DICOM', 'CT']:
        dcm_dir = os.path.join(parent, candidate)
        if os.path.isdir(dcm_dir):
            dcm_files = get_file_list(dcm_dir, '.dcm')
            if dcm_files:
                pixel_spacing, volume_position, patient_id = \
                    read_dcm_for_pixel_spacing(dcm_files)
                break

    return volume, volume_size, pixel_spacing, volume_position, patient_id


# ============================================================
# NIfTI Import (optional — requires nibabel)
# ============================================================

def _load_nifti(path, threshold=None, **kwargs):
    """Load NIfTI volume (.nii or .nii.gz)."""
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required for NIfTI support. "
            "Install with: pip install nibabel"
        )

    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine

    # Extract voxel spacing from affine
    pixel_spacing = np.abs(np.diag(affine[:3, :3])).astype(np.float32)
    volume_position = affine[:3, 3].astype(np.float32)
    patient_id = os.path.splitext(os.path.basename(path))[0]

    # Reorder to (D, H, W) if needed — NIfTI is typically (X, Y, Z)
    if data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))  # (Z, Y, X) = (D, H, W)
        pixel_spacing = pixel_spacing[[2, 1, 0]]

    print(f"NIfTI volume: {data.shape[2]}x{data.shape[1]}x{data.shape[0]}, "
          f"range: [{data.min():.1f}, {data.max():.1f}]", file=sys.stderr)

    # Threshold
    if threshold is None:
        # For pre-segmented NIfTI (label maps), use > 0
        unique_vals = np.unique(data)
        if len(unique_vals) <= 10:
            threshold = 0.0
            print(f"Label map detected ({len(unique_vals)} unique values), threshold=0",
                  file=sys.stderr)
        else:
            threshold = _otsu_threshold(data)
            print(f"Auto-threshold (Otsu): {threshold:.1f}", file=sys.stderr)

    mask_value = kwargs.get('mask_value', 255)
    volume = np.zeros_like(data, dtype=np.uint16)
    volume[data > threshold] = mask_value

    depth, height, width = volume.shape
    volume_size = (width, height, depth, 1)

    return volume, volume_size, pixel_spacing, volume_position, patient_id


# ============================================================
# NRRD Import (optional — requires pynrrd)
# ============================================================

def _load_nrrd(path, threshold=None, **kwargs):
    """Load NRRD volume (.nrrd)."""
    try:
        import nrrd
    except ImportError:
        raise ImportError(
            "pynrrd is required for NRRD support. "
            "Install with: pip install pynrrd"
        )

    data, header = nrrd.read(path)
    data = data.astype(np.float32)

    # Extract spacing
    pixel_spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if 'space directions' in header:
        dirs = np.array(header['space directions'])
        pixel_spacing = np.abs(np.diag(dirs)).astype(np.float32)
    elif 'spacings' in header:
        pixel_spacing = np.array(header['spacings'], dtype=np.float32)

    volume_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if 'space origin' in header:
        volume_position = np.array(header['space origin'], dtype=np.float32)

    patient_id = os.path.splitext(os.path.basename(path))[0]

    # Reorder to (D, H, W)
    if data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))
        pixel_spacing = pixel_spacing[[2, 1, 0]]

    print(f"NRRD volume: {data.shape[2]}x{data.shape[1]}x{data.shape[0]}, "
          f"range: [{data.min():.1f}, {data.max():.1f}]", file=sys.stderr)

    if threshold is None:
        unique_vals = np.unique(data)
        if len(unique_vals) <= 10:
            threshold = 0.0
        else:
            threshold = _otsu_threshold(data)
            print(f"Auto-threshold (Otsu): {threshold:.1f}", file=sys.stderr)

    mask_value = kwargs.get('mask_value', 255)
    volume = np.zeros_like(data, dtype=np.uint16)
    volume[data > threshold] = mask_value

    depth, height, width = volume.shape
    volume_size = (width, height, depth, 1)

    return volume, volume_size, pixel_spacing, volume_position, patient_id


# ============================================================
# VTP Export — VTK PolyData XML for ParaView
# ============================================================

def export_vtp(filename, points, faces=None, point_data=None):
    """Export mesh with data as VTK PolyData XML (.vtp) for ParaView.

    Args:
        filename: Output .vtp file path
        points: Nx3 array of vertex coordinates
        faces: Mx3 array of triangle indices (0-based), or None for point cloud
        point_data: dict of {name: Nx1 array} scalar fields per vertex
    """
    if not filename.endswith('.vtp'):
        filename += '.vtp'

    n_points = len(points)
    n_faces = len(faces) if faces is not None else 0

    # Encode arrays as base64 for compact VTP
    points_f32 = np.ascontiguousarray(points, dtype=np.float32)
    points_b64 = _encode_vtp_array(points_f32)

    with open(filename, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write('<PolyData>\n')

        if n_faces > 0:
            f.write(f'<Piece NumberOfPoints="{n_points}" NumberOfPolys="{n_faces}">\n')
        else:
            f.write(f'<Piece NumberOfPoints="{n_points}" NumberOfVerts="{n_points}">\n')

        # Point data (scalars like thickness)
        if point_data:
            first_name = list(point_data.keys())[0]
            f.write(f'<PointData Scalars="{first_name}">\n')
            for name, values in point_data.items():
                arr = np.ascontiguousarray(values, dtype=np.float32)
                b64 = _encode_vtp_array(arr)
                f.write(f'<DataArray type="Float32" Name="{name}" format="binary">\n')
                f.write(b64 + '\n')
                f.write('</DataArray>\n')
            f.write('</PointData>\n')

        # Points
        f.write('<Points>\n')
        f.write('<DataArray type="Float32" NumberOfComponents="3" format="binary">\n')
        f.write(points_b64 + '\n')
        f.write('</DataArray>\n')
        f.write('</Points>\n')

        if n_faces > 0:
            # Polys (triangles)
            connectivity = np.ascontiguousarray(faces.ravel(), dtype=np.int32)
            conn_b64 = _encode_vtp_array(connectivity)

            offsets = np.arange(3, 3 * n_faces + 1, 3, dtype=np.int32)
            off_b64 = _encode_vtp_array(offsets)

            f.write('<Polys>\n')
            f.write('<DataArray type="Int32" Name="connectivity" format="binary">\n')
            f.write(conn_b64 + '\n')
            f.write('</DataArray>\n')
            f.write('<DataArray type="Int32" Name="offsets" format="binary">\n')
            f.write(off_b64 + '\n')
            f.write('</DataArray>\n')
            f.write('</Polys>\n')
        else:
            # Vertices (point cloud)
            conn = np.arange(n_points, dtype=np.int32)
            conn_b64 = _encode_vtp_array(conn)
            offsets = np.arange(1, n_points + 1, dtype=np.int32)
            off_b64 = _encode_vtp_array(offsets)

            f.write('<Verts>\n')
            f.write('<DataArray type="Int32" Name="connectivity" format="binary">\n')
            f.write(conn_b64 + '\n')
            f.write('</DataArray>\n')
            f.write('<DataArray type="Int32" Name="offsets" format="binary">\n')
            f.write(off_b64 + '\n')
            f.write('</DataArray>\n')
            f.write('</Verts>\n')

        f.write('</Piece>\n')
        f.write('</PolyData>\n')
        f.write('</VTKFile>\n')

    print(f"VTP written: {filename} ({n_points} points, {n_faces} faces)")


def _encode_vtp_array(arr):
    """Encode numpy array as VTK binary base64 with header."""
    raw = arr.tobytes()
    # VTK binary format: 4-byte header with data length, then data
    header = struct.pack('<I', len(raw))
    return base64.b64encode(header + raw).decode('ascii')


# ============================================================
# PLT → VTP Converter (for existing results)
# ============================================================

def plt_to_vtp(plt_path, vtp_path=None):
    """Convert an existing PLT file to VTP for ParaView.

    Args:
        plt_path: Path to .plt file
        vtp_path: Output path (default: same name with .vtp extension)

    Returns:
        Path to output VTP file
    """
    if vtp_path is None:
        vtp_path = plt_path.rsplit('.', 1)[0] + '.vtp'

    points = []
    thickness = []

    with open(plt_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('VARIABLES') or line.startswith('ZONE'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                thickness.append(float(parts[3]))
            elif len(parts) == 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])

    points = np.array(points, dtype=np.float32)

    point_data = {}
    if thickness:
        point_data['Thickness(mm)'] = np.array(thickness, dtype=np.float32)

    # Check if PLT has connectivity (FEPOINT format)
    faces = _parse_plt_faces(plt_path)

    export_vtp(vtp_path, points, faces=faces, point_data=point_data)
    return vtp_path


def _parse_plt_faces(plt_path):
    """Parse triangle connectivity from PLT FEPOINT format."""
    faces = []
    in_data = False
    n_verts = 0
    n_elems = 0

    with open(plt_path) as f:
        for line in f:
            line = line.strip()
            if 'ET=triangle' in line.lower() or 'et=triangle' in line:
                # Parse N= and E= from ZONE line
                for token in line.replace(',', ' ').split():
                    if token.upper().startswith('N='):
                        n_verts = int(token[2:])
                    elif token.upper().startswith('E='):
                        n_elems = int(token[2:])
                in_data = True
                continue

            if in_data and n_verts > 0:
                n_verts -= 1
                if n_verts == 0:
                    in_data = False
                    # Next lines are faces
                    for face_line in f:
                        parts = face_line.strip().split()
                        if len(parts) >= 3:
                            # PLT uses 1-based indexing
                            faces.append([int(parts[0])-1, int(parts[1])-1, int(parts[2])-1])
                            if len(faces) >= n_elems:
                                break
                    break

    if faces:
        return np.array(faces, dtype=np.int32)
    return None


# ============================================================
# STL → VTP Converter
# ============================================================

def stl_to_vtp(stl_path, vtp_path=None, thickness_plt=None):
    """Convert ASCII STL to VTP, optionally mapping thickness data.

    Args:
        stl_path: Path to ASCII .stl file
        vtp_path: Output path
        thickness_plt: Optional PLT file with thickness data for mapping
    """
    if vtp_path is None:
        vtp_path = stl_path.rsplit('.', 1)[0] + '.vtp'

    # Parse ASCII STL
    vertices = []
    normals = []

    with open(stl_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('vertex'):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('facet normal'):
                parts = line.split()
                normals.append([float(parts[2]), float(parts[3]), float(parts[4])])

    if not vertices:
        print(f"No vertices found in {stl_path}", file=sys.stderr)
        return

    vertices = np.array(vertices, dtype=np.float32)
    n_tris = len(vertices) // 3

    # Deduplicate vertices
    unique_verts, inverse = _deduplicate_vertices(vertices)
    faces = inverse.reshape(-1, 3)

    # Map thickness if PLT provided
    point_data = {}
    if thickness_plt and os.path.exists(thickness_plt):
        point_data = _map_thickness_from_plt(unique_verts, thickness_plt)

    export_vtp(vtp_path, unique_verts, faces=faces, point_data=point_data)
    return vtp_path


def _deduplicate_vertices(vertices, decimals=4):
    """Deduplicate vertices by rounding."""
    rounded = np.round(vertices, decimals)
    unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
    return unique.astype(np.float32), inverse.astype(np.int32)


def _map_thickness_from_plt(vertices, plt_path):
    """Map thickness values from PLT to mesh vertices using KD-tree."""
    from scipy.spatial import cKDTree

    plt_points = []
    plt_thickness = []

    with open(plt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    plt_points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    plt_thickness.append(float(parts[3]))
                except ValueError:
                    continue

    if not plt_points:
        return {}

    plt_points = np.array(plt_points, dtype=np.float32)
    plt_thickness = np.array(plt_thickness, dtype=np.float32)

    tree = cKDTree(plt_points)
    _, indices = tree.query(vertices, k=1)
    thickness = plt_thickness[indices]

    return {'Thickness(mm)': thickness}
