"""AutoAWT - Automatic Atrial Wall Thickness Measurement
Python port of the C++/CUDA application.

Usage:
    # Mode 1: BMP masks + DICOM metadata (original pipeline)
    python main.py --masks <bmp_directory> --dicom <dcm_directory>

    # Mode 2: Direct DICOM import (segment from CT volume)
    python main.py --input <dicom_directory> --threshold <HU_value>

    # Mode 3: NIfTI / NRRD (pre-segmented or raw volume)
    python main.py --input <file.nii.gz> --threshold 0

    # Interactive (GUI file dialogs)
    python main.py

Options:
    --input PATH      Input volume (DICOM dir, .nii, .nii.gz, .nrrd)
                      Auto-detects format. For DICOM, reads pixel data directly.
    --masks PATH      BMP mask directory (legacy mode)
    --dicom PATH      DICOM directory for metadata only (legacy mode)
    --threshold N     HU/intensity threshold for binary mask (auto if omitted)
    --output PATH     Output directory (default: Results/ under input dir)
    --format FMT      Force input format: dicom, bmp, nifti, nrrd
    --paraview        Export VTP files for ParaView visualization

Produces:
    - Results/WT-endo.plt : Wall thickness from endocardium to epicardium
    - Results/WT(projected)-PatientID.plt : Thickness projected to surface mesh
    - Results/WT(projected)-PatientID.stl : Surface mesh
    - Results/surface_mesh.stl : Reconstructed surface mesh
    - Results/Epi-Endo/ : Labeled images (1=endo, 2=myocardium, 3=epi)
    - Results/*.vtp : ParaView files (if --paraview)
"""

import os
import sys
import time
import argparse
import numpy as np

from utils import (
    get_file_list, select_directory, read_bmp_files,
    read_dcm_for_pixel_spacing, export_bmp, compute_fill_space,
    arr_logical_and
)
from qhull import QHull, CTview
from wt import WT
from marching_cubes import MarchingCube


def run_pipeline(g_wall_mask, volume_size, g_pixel_spacing, volume_position,
                 patient_id, data_path, output_path=None, export_paraview=False):
    """Run the full AutoAWT pipeline on a loaded volume.

    Args:
        g_wall_mask: uint16 3D array (D, H, W) — binary wall mask (0/255)
        volume_size: (width, height, depth, channels)
        g_pixel_spacing: [sx, sy, sz] in mm
        volume_position: [x0, y0, z0] in mm
        patient_id: string
        data_path: path for intermediate BMP exports
        output_path: Results directory (default: data_path/Results)
        export_paraview: if True, export .vtp files for ParaView

    Returns:
        0 on success, 1 on error
    """
    w, h, d = volume_size[0], volume_size[1], volume_size[2]

    print(f"Volume: {w}x{h}x{d}")
    print(f"Voxel spacing: {g_pixel_spacing[0]:.4f}, {g_pixel_spacing[1]:.4f}, {g_pixel_spacing[2]:.4f} mm")
    print(f"Volume position: {volume_position[0]:.2f}, {volume_position[1]:.2f}, {volume_position[2]:.2f}")
    print(f"Patient ID: {patient_id}")

    # Step 1: Convex Hull Generation
    merged_hull = np.zeros((d, h, w), dtype=np.uint16)
    hull_mask = np.zeros((d, h, w), dtype=np.uint16)

    # Axial convex hulls
    sum_time = 0.0
    print("Computing axial convex hulls...")
    for i in range(d):
        st = time.time()
        qh = QHull()
        qh.set_point_cloud(g_wall_mask, volume_size, i, CTview.AXIAL)
        if len(qh.get_point_cloud()) < 1:
            continue

        qh.initialize()
        convex_points = qh.get_drawable_points()
        sum_time += (time.time() - st) * 1000
        compute_fill_space(merged_hull, convex_points, i, (w, h, d), CTview.AXIAL)

    print(f"  Axial hulls: {sum_time:.0f}ms")
    export_bmp(merged_hull, volume_size, "axial", data_path)

    # Sagittal convex hulls
    sum_time = 0.0
    print("Computing sagittal convex hulls...")
    for i in range(w):
        st = time.time()
        qh = QHull()
        qh.set_point_cloud(g_wall_mask, volume_size, i, CTview.SAGITTAL)
        if len(qh.get_point_cloud()) < 1:
            continue

        qh.initialize()
        convex_points = qh.get_drawable_points()
        sum_time += (time.time() - st) * 1000
        compute_fill_space(hull_mask, convex_points, i, (w, h, d), CTview.SAGITTAL)

    arr_logical_and(merged_hull, hull_mask)
    print(f"  Sagittal hulls: {sum_time:.0f}ms")

    # Coronal convex hulls
    sum_time = 0.0
    hull_mask[:] = 0
    print("Computing coronal convex hulls...")
    for i in range(h):
        st = time.time()
        qh = QHull()
        qh.set_point_cloud(g_wall_mask, volume_size, i, CTview.CORONAL)
        if len(qh.get_point_cloud()) < 1:
            continue

        qh.initialize()
        convex_points = qh.get_drawable_points()
        sum_time += (time.time() - st) * 1000
        compute_fill_space(hull_mask, convex_points, i, (w, h, d), CTview.CORONAL)

    arr_logical_and(merged_hull, hull_mask)
    print(f"  Coronal hulls: {sum_time:.0f}ms")

    export_bmp(merged_hull, volume_size, "Merged-hull", data_path)
    del hull_mask

    # Step 2: Wall Thickness Computation
    if output_path is None:
        output_path = os.path.join(data_path, "Results")
    os.makedirs(output_path, exist_ok=True)

    st_time = time.time()

    wt_algorithms = WT(
        output_path, volume_size, g_pixel_spacing,
        volume_position, g_wall_mask, merged_hull
    )
    wt_algorithms.detect_epi_endo(None)

    calc_time = (time.time() - st_time) * 1000
    print(f"Epi-endo detection: {calc_time:.0f}ms ({calc_time / 1e3:.1f}s)", file=sys.stderr)

    st_time = time.time()
    wt_algorithms.eval_wt()
    calc_time = (time.time() - st_time) * 1000
    print(f"Wall thickness: {calc_time:.0f}ms ({calc_time / 1e3:.1f}s)", file=sys.stderr)

    # Copy chamber mask into g_wall_mask (reuse array for Marching Cubes input)
    chamber_mask = wt_algorithms.get_chamber_mask()
    endo_vertices_list = list(wt_algorithms.m_endo_vertices_list)
    del wt_algorithms

    # Step 3: Surface Mesh Generation (Marching Cubes) — uses chamber mask
    surfaces = MarchingCube(
        output_path, chamber_mask, volume_size,
        g_pixel_spacing, volume_position, MarchingCube.FROM_HOST
    )
    surfaces.compute_isosurface(chamber_mask, MarchingCube.FROM_HOST)
    surfaces.save_mesh_info(
        os.path.join(output_path, f"WT(projected)-{patient_id}"),
        endo_vertices_list
    )
    del chamber_mask, merged_hull

    # Step 4: ParaView export
    if export_paraview:
        _export_paraview_files(output_path, patient_id)

    print("AutoAWT processing complete.")
    print(f"Results saved to: {output_path}")
    return 0


def _export_paraview_files(output_path, patient_id):
    """Export VTP files for ParaView visualization."""
    from io_formats import plt_to_vtp, stl_to_vtp

    print("Exporting VTP files for ParaView...")

    # Convert endo thickness point cloud
    endo_plt = os.path.join(output_path, "WT-endo.plt")
    if os.path.exists(endo_plt):
        vtp = plt_to_vtp(endo_plt)
        print(f"  {vtp}")

    # Convert projected thickness mesh
    proj_plt = None
    for f in os.listdir(output_path):
        if f.startswith("WT(projected)") and f.endswith(".plt"):
            proj_plt = os.path.join(output_path, f)
            break
    if proj_plt:
        vtp = plt_to_vtp(proj_plt)
        print(f"  {vtp}")

    # Convert surface mesh STL with thickness mapping
    stl_file = os.path.join(output_path, "surface_mesh.stl")
    if os.path.exists(stl_file):
        vtp = stl_to_vtp(stl_file, thickness_plt=endo_plt if os.path.exists(endo_plt) else None)
        print(f"  {vtp}")


def main():
    parser = argparse.ArgumentParser(
        description='AutoAWT - Automatic Atrial Wall Thickness Measurement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Legacy mode: BMP masks + DICOM metadata
  python main.py --masks /path/to/bmp_masks --dicom /path/to/dicom

  # Direct DICOM import with auto-threshold
  python main.py --input /path/to/dicom_series

  # Direct DICOM import with manual threshold
  python main.py --input /path/to/dicom_series --threshold -200

  # NIfTI segmentation mask
  python main.py --input segmentation.nii.gz --threshold 0

  # With ParaView export
  python main.py --input /path/to/data --paraview
        """
    )

    # New multi-format input
    parser.add_argument('--input', type=str,
                        help='Input volume: DICOM dir, .nii(.gz), .nrrd, or BMP dir')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Intensity threshold for mask (auto-detected if omitted)')
    parser.add_argument('--format', type=str, default=None,
                        choices=['dicom', 'bmp', 'nifti', 'nrrd'],
                        help='Force input format (auto-detected if omitted)')

    # Legacy mode (backwards-compatible)
    parser.add_argument('--masks', type=str,
                        help='BMP mask directory (legacy mode)')
    parser.add_argument('--dicom', type=str,
                        help='DICOM directory for pixel spacing (legacy mode)')

    # Output options
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: Results/ under input)')
    parser.add_argument('--paraview', action='store_true',
                        help='Export VTP files for ParaView visualization')

    args = parser.parse_args()

    # Determine which mode to use
    if args.input:
        # New multi-format mode
        return _run_input_mode(args)
    elif args.masks:
        # Legacy BMP + DICOM mode
        return _run_legacy_mode(args)
    else:
        # Interactive — ask user which mode
        return _run_interactive_mode(args)


def _run_input_mode(args):
    """New multi-format input mode."""
    from io_formats import load_volume

    print(f"Loading volume from: {args.input}")
    volume, volume_size, pixel_spacing, volume_position, patient_id = \
        load_volume(args.input, fmt=args.format, threshold=args.threshold)

    # Determine data_path for intermediate files
    if os.path.isdir(args.input):
        data_path = args.input
    else:
        data_path = os.path.dirname(args.input)

    return run_pipeline(
        volume, volume_size, pixel_spacing, volume_position, patient_id,
        data_path, output_path=args.output, export_paraview=args.paraview
    )


def _run_legacy_mode(args):
    """Legacy BMP masks + DICOM metadata mode."""
    bmp_path = args.masks

    if not bmp_path or not os.path.isdir(bmp_path):
        print("Invalid BMP directory", file=sys.stderr)
        return 1

    file_names = get_file_list(bmp_path, '.bmp')
    print(f"Found {len(file_names)} BMP files")

    if len(file_names) == 0:
        print("No BMP files found in directory", file=sys.stderr)
        return 1

    g_wall_mask, volume_size = read_bmp_files(file_names)
    if g_wall_mask is None:
        return 1

    # DICOM metadata
    if args.dicom:
        dcm_path = args.dicom
    else:
        dcm_path = select_directory("DICOM selection path pixel spacing (dcm)")

    if not dcm_path or not os.path.isdir(dcm_path):
        print("Invalid DICOM directory", file=sys.stderr)
        return 1

    dcm_files = get_file_list(dcm_path, '.dcm')
    print(f"Found {len(dcm_files)} DICOM files")

    if len(dcm_files) == 0:
        print("No DICOM files found in directory", file=sys.stderr)
        return 1

    g_pixel_spacing, volume_position, patient_id = read_dcm_for_pixel_spacing(dcm_files)

    return run_pipeline(
        g_wall_mask, volume_size, g_pixel_spacing, volume_position, patient_id,
        bmp_path, output_path=args.output, export_paraview=args.paraview
    )


def _run_interactive_mode(args):
    """Interactive mode with GUI dialogs."""
    print("No input specified. Select input mode:")
    print("  1. BMP masks + DICOM metadata (legacy)")
    print("  2. DICOM volume (direct import)")
    print("  3. NIfTI/NRRD file")

    try:
        choice = input("Choice [1/2/3]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return 1

    if choice == '2':
        path = select_directory("Select DICOM series directory")
        if not path:
            return 1
        threshold_str = input("HU threshold (Enter for auto): ").strip()
        try:
            threshold = float(threshold_str) if threshold_str else None
        except ValueError:
            print(f"Invalid threshold '{threshold_str}', using auto-detection", file=sys.stderr)
            threshold = None
        args.input = path
        args.threshold = threshold
        args.format = 'dicom'
        return _run_input_mode(args)

    elif choice == '3':
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title="Select NIfTI/NRRD file",
                filetypes=[("NIfTI", "*.nii *.nii.gz"), ("NRRD", "*.nrrd"), ("All", "*.*")]
            )
            root.destroy()
        except ImportError:
            path = input("Enter file path: ").strip()
        if not path:
            return 1
        threshold_str = input("Threshold (Enter for auto): ").strip()
        try:
            threshold = float(threshold_str) if threshold_str else None
        except ValueError:
            print(f"Invalid threshold '{threshold_str}', using auto-detection", file=sys.stderr)
            threshold = None
        args.input = path
        args.threshold = threshold
        return _run_input_mode(args)

    else:
        # Legacy BMP mode
        bmp_path = select_directory("Wall mask selection path (BMP)")
        if not bmp_path:
            return 1
        args.masks = bmp_path
        args.dicom = None
        return _run_legacy_mode(args)


if __name__ == '__main__':
    sys.exit(main())
