"""Marching Cubes isosurface extraction.
Ported from MarchingCubes.cpp/h and CUDA kernels in marchingCube_kernel.cu/cuh

Optimized with NumPy vectorization for CPU, CuPy for GPU.
"""

import os
import sys
import math
import time
import numpy as np
from scipy import ndimage

from backend import xp, to_numpy, to_device, GPU_AVAILABLE
from mc_tables import edgeTable, triTable, numVertsTable


def roundf_digit(num, d):
    """Round float to d-1 decimal places (floor).
    Port of roundf_digit from MarchingCubes.cpp:51
    """
    t = 10 ** (d - 1)
    return math.floor(num * t) / t


# Pre-convert tables to numpy arrays for fast indexing
_numVertsTable = np.array(numVertsTable, dtype=np.uint32)
_triTable = np.array(triTable, dtype=np.int32)  # int32 for -1/255 sentinel checking

# Edge pairs for the 12 edges of a cube
_EDGE_PAIRS = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
], dtype=np.int32)

# 8 cube vertex offsets (relative to base corner)
_CUBE_OFFSETS = np.array([
    [0, 0, 0],  # v0
    [1, 0, 0],  # v1
    [1, 1, 0],  # v2
    [0, 1, 0],  # v3
    [0, 0, 1],  # v4
    [1, 0, 1],  # v5
    [1, 1, 1],  # v6
    [0, 1, 1],  # v7
], dtype=np.float32)


class MarchingCube:
    """Marching Cubes isosurface extraction.
    Exact port of MarchingCube class from MarchingCubes.cpp/h
    """

    FROM_HOST = 0
    FROM_DEVICE = 1

    def __init__(self, save_path, volume, volume_size, pixel_spacing, volume_center, input_type=0):
        self.m_save_path = save_path

        if len(volume_size) == 4:
            self.m_volume_size = np.array([volume_size[0], volume_size[1], volume_size[2], volume_size[3]], dtype=np.float32)
        else:
            self.m_volume_size = np.array([volume_size[0], volume_size[1], volume_size[2], 1.0], dtype=np.float32)

        w = int(self.m_volume_size[0])
        h = int(self.m_volume_size[1])
        d = int(self.m_volume_size[2])

        self.m_num_voxels = w * h * d
        self.m_max_vertices = self.m_num_voxels
        self.m_voxel_size = pixel_spacing.copy().astype(np.float32)
        self.m_voxel_center = volume_center.copy().astype(np.float32)
        self.m_total_verts = 0

        self.m_d_volume = np.zeros((d, h, w), dtype=np.uint16)
        self.d_pos = None
        self.d_normal = None

    def compute_isosurface(self, volume=None, input_type=0):
        """Compute isosurface using Marching Cubes algorithm."""
        st_time = time.time()

        iso_value = 0.5
        w = int(self.m_volume_size[0])
        h = int(self.m_volume_size[1])
        d = int(self.m_volume_size[2])

        if volume is not None:
            if input_type == self.FROM_HOST:
                self.m_d_volume = volume.copy().astype(np.uint16)
                self._inverse_depth_volume()
            else:
                self.m_d_volume = volume.copy().astype(np.uint16)
                self._median_filter_3d(kernel_size=3)
                self._inverse_depth_volume()

        # Classify voxels (fully vectorized)
        voxel_verts, cube_index_3d = self._classify_voxels(iso_value)

        # Prefix sum (exclusive scan)
        voxel_verts_scan = np.zeros_like(voxel_verts)
        if len(voxel_verts) > 0:
            voxel_verts_scan[1:] = np.cumsum(voxel_verts[:-1])

        # Total vertices
        if len(voxel_verts) > 0:
            last_elem = int(voxel_verts[-1])
            last_scan = int(voxel_verts_scan[-1])
            self.m_total_verts = last_elem + last_scan
        else:
            self.m_total_verts = 0

        # Allocate output arrays (only what we need + some margin)
        alloc_size = max(self.m_total_verts + 16, 16)
        self.d_pos = np.zeros((alloc_size, 4), dtype=np.float32)
        self.d_normal = np.zeros((alloc_size, 4), dtype=np.float32)

        # Generate triangles (optimized: only active voxels)
        self._gen_triangles(voxel_verts_scan, cube_index_3d, iso_value)

        ed_time = time.time()
        calc_time = (ed_time - st_time) * 1000

        # Save surface mesh
        st_time = time.time()
        self.save_mesh_info(os.path.join(self.m_save_path, "surface_mesh"))
        ed_time = time.time()

        print(f"create surfaces = {self.m_total_verts}", file=sys.stderr)
        print(f"{calc_time:.0f}ms, write file = {(ed_time - st_time) * 1000:.0f}ms", file=sys.stderr)

    def _inverse_depth_volume(self):
        """Reverse Z-axis of volume.
        Port of inverse_depth_volume kernel from marchingCube_kernel.cuh:213-231
        """
        self.m_d_volume = np.flip(self.m_d_volume, axis=0).copy()

    def _median_filter_3d(self, kernel_size=3):
        """Apply 3D median filter."""
        self.m_d_volume = ndimage.median_filter(
            self.m_d_volume, size=kernel_size
        ).astype(np.uint16)

    def _classify_voxels(self, iso_value):
        """Classify voxels - fully vectorized.
        Port of classifyVoxel kernel from marchingCube_kernel.cuh:76-111
        """
        w = int(self.m_volume_size[0])
        h = int(self.m_volume_size[1])
        d = int(self.m_volume_size[2])

        vol = self.m_d_volume.astype(np.float32)

        # Pad volume by 1 on positive sides (edge clamping)
        padded = np.pad(vol, ((0, 1), (0, 1), (0, 1)), mode='edge')

        # Sample 8 corners for all voxels simultaneously
        # volume[z,y,x] -> padded slicing gives us neighbor access
        f0 = padded[:-1, :-1, :-1]  # (tx, ty, tz)
        f1 = padded[:-1, :-1, 1:]   # (tx+1, ty, tz)
        f2 = padded[:-1, 1:, 1:]    # (tx+1, ty+1, tz)
        f3 = padded[:-1, 1:, :-1]   # (tx, ty+1, tz)
        f4 = padded[1:, :-1, :-1]   # (tx, ty, tz+1)
        f5 = padded[1:, :-1, 1:]    # (tx+1, ty, tz+1)
        f6 = padded[1:, 1:, 1:]     # (tx+1, ty+1, tz+1)
        f7 = padded[1:, 1:, :-1]    # (tx, ty+1, tz+1)

        # Compute cube index as bitfield
        cube_index = ((f0 > 0).astype(np.uint32) |
                      ((f1 > 0).astype(np.uint32) << 1) |
                      ((f2 > 0).astype(np.uint32) << 2) |
                      ((f3 > 0).astype(np.uint32) << 3) |
                      ((f4 > 0).astype(np.uint32) << 4) |
                      ((f5 > 0).astype(np.uint32) << 5) |
                      ((f6 > 0).astype(np.uint32) << 6) |
                      ((f7 > 0).astype(np.uint32) << 7))

        # Lookup numVertsTable for each voxel
        voxel_verts_3d = _numVertsTable[cube_index]

        # Flatten with C order: z*h*w + y*w + x (matches idx = tx + ty*w + tz*w*h)
        voxel_verts = voxel_verts_3d.ravel(order='C')

        return voxel_verts, cube_index

    def _gen_triangles(self, num_verts_scanned, cube_index_3d, iso_value):
        """Generate triangle vertices - optimized to only process active voxels.
        Port of genTriangles kernel from marchingCube_kernel.cuh:114-211
        """
        w = int(self.m_volume_size[0])
        h = int(self.m_volume_size[1])
        d = int(self.m_volume_size[2])

        voxel_size = self.m_voxel_size
        voxel_center = self.m_voxel_center
        output_size = len(self.d_pos)

        # Find active voxels (voxels that produce vertices)
        num_verts_3d = _numVertsTable[cube_index_3d]
        active_mask = num_verts_3d > 0
        active_indices = np.argwhere(active_mask)  # (N, 3) = (z, y, x)
        n_active = len(active_indices)

        print(f"  Active voxels: {n_active} / {w * h * d}", file=sys.stderr)

        if n_active == 0:
            return

        # Active voxel coordinates
        az = active_indices[:, 0]
        ay = active_indices[:, 1]
        ax = active_indices[:, 2]

        # Pre-compute physical positions for all active voxels (N, 3)
        txs = ax.astype(np.float32)
        tys = ay.astype(np.float32)
        tzs = az.astype(np.float32)

        base_pos = np.column_stack([
            voxel_center[0] + txs * voxel_size[0],
            voxel_center[1] + tys * voxel_size[1],
            voxel_center[2] - d * voxel_size[2] + tzs * voxel_size[2]
        ])  # (N, 3)

        # Cube vertex offsets scaled by voxel_size
        scaled_offsets = _CUBE_OFFSETS * voxel_size  # (8, 3)

        # Pre-compute all 12 edge midpoints for all active voxels (N, 12, 3)
        # C++ vertexInterp computes t but uses lerp(p0, p1, isoValue) = midpoint
        vertlists = np.zeros((n_active, 12, 3), dtype=np.float32)
        for ei in range(12):
            a, b = _EDGE_PAIRS[ei]
            va = base_pos + scaled_offsets[a]  # (N, 3)
            vb = base_pos + scaled_offsets[b]  # (N, 3)
            vertlists[:, ei, :] = va + iso_value * (vb - va)

        # Get cube indices and num_verts for active voxels
        active_cube_idx = cube_index_3d[active_mask].ravel()
        active_num_verts = num_verts_3d[active_mask].ravel()

        # Compute linear indices for scan lookup (must match C-order ravel)
        linear_indices = (ax.astype(np.int64) +
                          ay.astype(np.int64) * w +
                          az.astype(np.int64) * w * h)

        # Process each active voxel
        for i_vox in range(n_active):
            li = int(linear_indices[i_vox])

            ci = int(active_cube_idx[i_vox])
            num_v = int(active_num_verts[i_vox])

            scan_offset = int(num_verts_scanned[li])

            for i in range(0, num_v, 3):
                out_index = scan_offset + i

                e0 = int(_triTable[ci, i])
                e1 = int(_triTable[ci, i + 2])  # swapped order (C++ code)
                e2 = int(_triTable[ci, i + 1])

                if e0 >= 12 or e1 >= 12 or e2 >= 12 or e0 < 0 or e1 < 0 or e2 < 0:
                    continue

                v0 = vertlists[i_vox, e0]
                v1 = vertlists[i_vox, e1]
                v2 = vertlists[i_vox, e2]

                # Normal via cross product
                edge0 = v1 - v0
                edge1 = v2 - v0
                nx = edge0[1] * edge1[2] - edge0[2] * edge1[1]
                ny = edge0[2] * edge1[0] - edge0[0] * edge1[2]
                nz = edge0[0] * edge1[1] - edge0[1] * edge1[0]

                if out_index < output_size - 3:
                    self.d_pos[out_index] = [v0[0], v0[1], v0[2], 1.0]
                    self.d_normal[out_index] = [nx, ny, nz, 0.0]
                    self.d_pos[out_index + 1] = [v1[0], v1[1], v1[2], 1.0]
                    self.d_normal[out_index + 1] = [nx, ny, nz, 0.0]
                    self.d_pos[out_index + 2] = [v2[0], v2[1], v2[2], 1.0]
                    self.d_normal[out_index + 2] = [nx, ny, nz, 0.0]

            if i_vox % 50000 == 0 and i_vox > 0:
                print(f"  gen_triangles: {i_vox}/{n_active}", file=sys.stderr)

    def save_mesh_info(self, filename, vertices_list=None):
        """Save mesh to STL and optionally PLT with thickness.
        Port of MarchingCube::saveMeshInfo() from MarchingCubes.cpp:334-348
        """
        self._write_stl(filename)

        if vertices_list is None:
            return
        if len(vertices_list) == 0:
            return

        self._write_plt(filename, vertices_list)

    def _write_stl(self, filename):
        """Write mesh in ASCII STL format.
        Optimized to only iterate up to m_total_verts.
        """
        total = self.m_total_verts

        with open(filename + ".stl", 'w') as f:
            f.write("solid ascii \n")

            for i in range(0, total - 2, 3):
                pos = self.d_pos[i]
                if pos[3] == 0:
                    continue

                normal = self.d_normal[i]
                f.write(f"facet normal {normal[0]} {normal[1]} {normal[2]} \n")
                f.write("outer loop\n")

                f.write(f"vertex {pos[0]} {pos[1]} {pos[2]} \n")
                pos1 = self.d_pos[i + 1]
                f.write(f"vertex {pos1[0]} {pos1[1]} {pos1[2]} \n")
                pos2 = self.d_pos[i + 2]
                f.write(f"vertex {pos2[0]} {pos2[1]} {pos2[2]} \n")

                f.write("endloop\n")
                f.write("endfacet\n")

            f.write("endsolid \n")

        print("stl file write complete", file=sys.stderr)

    def _write_plt(self, filename, vertices_list):
        """Write mesh in Tecplot PLT format with thickness mapping.
        Uses scipy.spatial.cKDTree for fast nearest-neighbor lookup.
        """
        from scipy.spatial import cKDTree

        total = self.m_total_verts
        h_pos = self.d_pos[:total].copy()
        h_normal = self.d_normal[:total].copy()

        # Round vertex coordinates
        for i in range(total):
            h_pos[i, 0] = roundf_digit(h_pos[i, 0], 3)
            h_pos[i, 1] = roundf_digit(h_pos[i, 1], 3)
            h_pos[i, 2] = roundf_digit(h_pos[i, 2], 3)

        # Build vertex map with deduplication
        vertices_map = {}
        num_indices = 0

        for i in range(total):
            if h_pos[i, 3] == 0:
                continue
            vert_key = (h_pos[i, 0], h_pos[i, 1], h_pos[i, 2])
            num_indices += 1
            if vert_key not in vertices_map:
                vertices_map[vert_key] = len(vertices_map) + 1

        # Sort by insertion order (index)
        order = sorted(vertices_map.items(), key=lambda x: x[1])

        # Build KD-tree for endo vertices (fast nearest-neighbor)
        if len(vertices_list) > 0:
            endo_pts = np.array([[v[0], v[1], v[2]] for v in vertices_list], dtype=np.float32)
            endo_w = np.array([v[3] for v in vertices_list], dtype=np.float32)
            tree = cKDTree(endo_pts)
        else:
            tree = None
            endo_w = None

        # Write PLT file
        with open(filename + ".plt", 'w') as f:
            f.write('VARIABLES = "X", "Y", "Z", "Thickness"\n')
            f.write(f'ZONE F=FEPOINT, ET=triangle, N={len(vertices_map)} , E={num_indices // 3}\n')

            print(f"# of vert: {len(vertices_map)}")

            for vert_key, vert_idx in order:
                vert = np.array(vert_key, dtype=np.float32)

                if tree is not None:
                    dist, nn_idx = tree.query(vert)
                    closest_w = float(endo_w[nn_idx])
                else:
                    closest_w = 0.0

                f.write(f"{vert[0]} {vert[1]} {vert[2]} {closest_w}\n")

            # Write triangle connectivity
            num_faces = 0
            for i in range(0, total, 3):
                if h_pos[i, 3] == 0:
                    continue

                v1_key = (h_pos[i, 0], h_pos[i, 1], h_pos[i, 2])
                v2_key = (h_pos[i + 1, 0], h_pos[i + 1, 1], h_pos[i + 1, 2])
                v3_key = (h_pos[i + 2, 0], h_pos[i + 2, 1], h_pos[i + 2, 2])

                idx1 = vertices_map.get(v1_key, 0)
                idx2 = vertices_map.get(v2_key, 0)
                idx3 = vertices_map.get(v3_key, 0)

                if idx1 == 0 or idx2 == 0 or idx3 == 0:
                    continue

                num_faces += 1
                f.write(f"{idx1} {idx2} {idx3}\n")

        # Rewrite header with correct face count
        with open(filename + ".plt", 'r') as f:
            lines = f.readlines()

        lines[1] = f'ZONE F=FEPOINT, ET=triangle, N={len(vertices_map)} , E={num_faces}\n'

        with open(filename + ".plt", 'w') as f:
            f.writelines(lines)

        print("plt file write complete", file=sys.stderr)
        print(filename + ".plt")
