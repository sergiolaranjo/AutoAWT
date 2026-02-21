"""AutoAWT - Automatic Atrial Wall Thickness Measurement
Python port of the C++/CUDA application.

Usage:
    python main.py --masks <bmp_directory> --dicom <dcm_directory>
    python main.py  (uses GUI file dialogs)

Produces the same results as the original C++ implementation:
    - Results/WT-endo.plt : Wall thickness from endocardium to epicardium
    - Results/WT(projected)-PatientID.plt : Thickness projected to surface mesh
    - Results/WT(projected)-PatientID.stl : Surface mesh with thickness
    - Results/surface_mesh.stl : Reconstructed surface mesh
    - Results/Epi-Endo/ : Labeled images (1=endo, 2=myocardium, 3=epi)
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


def main():
    parser = argparse.ArgumentParser(description='AutoAWT - Automatic Atrial Wall Thickness Measurement')
    parser.add_argument('--masks', type=str, help='Path to BMP mask images directory')
    parser.add_argument('--dicom', type=str, help='Path to DICOM files directory')
    args = parser.parse_args()

    # Step 1: Select BMP mask directory
    if args.masks:
        bmp_path = args.masks
    else:
        bmp_path = select_directory("Wall mask selection path (BMP)")

    if not bmp_path or not os.path.isdir(bmp_path):
        print("Invalid BMP directory", file=sys.stderr)
        return 1

    file_names = get_file_list(bmp_path, '.bmp')
    print(f"Found {len(file_names)} files")

    g_wall_mask, volume_size = read_bmp_files(file_names)
    if g_wall_mask is None:
        return 1

    # Step 2: Select DICOM directory
    if args.dicom:
        dcm_path = args.dicom
    else:
        dcm_path = select_directory("DICOM selection path pixel spacing (dcm)")

    if not dcm_path or not os.path.isdir(dcm_path):
        print("Invalid DICOM directory", file=sys.stderr)
        return 1

    dcm_files = get_file_list(dcm_path, '.dcm')
    print(f"Found {len(dcm_files)} files")

    g_pixel_spacing, volume_position, patient_id = read_dcm_for_pixel_spacing(dcm_files)
    print(f"voxel spacing = {g_pixel_spacing[0]}, {g_pixel_spacing[1]}, {g_pixel_spacing[2]}")
    print(f"volume_position = {volume_position[0]}, {volume_position[1]}, {volume_position[2]}")

    w, h, d = volume_size[0], volume_size[1], volume_size[2]

    # Step 3: Convex Hull Generation
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

        ed = time.time()
        dt = (ed - st) * 1000
        sum_time += dt

        compute_fill_space(merged_hull, convex_points, i, (w, h, d), CTview.AXIAL)

    print(f"acc_time: {sum_time}")
    export_bmp(merged_hull, volume_size, "axial", bmp_path)

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

        ed = time.time()
        dt = (ed - st) * 1000
        sum_time += dt

        compute_fill_space(hull_mask, convex_points, i, (w, h, d), CTview.SAGITTAL)

    arr_logical_and(merged_hull, hull_mask)
    print(f"acc_time: {sum_time}")

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

        ed = time.time()
        dt = (ed - st) * 1000
        sum_time += dt

        compute_fill_space(hull_mask, convex_points, i, (w, h, d), CTview.CORONAL)

    arr_logical_and(merged_hull, hull_mask)
    print(f"acc_time: {sum_time}")

    export_bmp(merged_hull, volume_size, "Merged-hull", bmp_path)
    del hull_mask

    # Step 4: Wall Thickness Computation
    results_path = os.path.join(bmp_path, "Results")
    os.makedirs(results_path, exist_ok=True)

    st_time = time.time()

    wt_algorithms = WT(
        results_path, volume_size, g_pixel_spacing,
        volume_position, g_wall_mask, merged_hull
    )
    wt_algorithms.detect_epi_endo(None)

    ed_time = time.time()
    calc_time = (ed_time - st_time) * 1000
    print(f"Finished epi-endo calculation : {calc_time:.0f}ms ({calc_time / 1e3:.1f} s)", file=sys.stderr)

    st_time = time.time()
    wt_algorithms.eval_wt()
    ed_time = time.time()
    calc_time = (ed_time - st_time) * 1000
    print(f"Finished WT : {calc_time:.0f}ms ({calc_time / 1e3:.1f} s)", file=sys.stderr)

    # Copy results
    g_wall_mask[:] = wt_algorithms.get_chamber_mask()
    endo_vertices_list = list(wt_algorithms.m_endo_vertices_list)
    del wt_algorithms

    # Step 5: Surface Mesh Generation (Marching Cubes)
    surfaces = MarchingCube(
        results_path, g_wall_mask, volume_size,
        g_pixel_spacing, volume_position, MarchingCube.FROM_HOST
    )
    surfaces.compute_isosurface(g_wall_mask, MarchingCube.FROM_HOST)
    surfaces.save_mesh_info(
        os.path.join(results_path, f"WT(projected)-{patient_id}"),
        endo_vertices_list
    )
    del merged_hull

    print("AutoAWT processing complete.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
