"""2D Convex Hull using Jarvis March (Gift Wrapping) algorithm.
Ported from QHull.cpp/h
"""

import numpy as np
from enum import IntEnum


class CTview(IntEnum):
    AXIAL = 0
    CORONAL = 1
    SAGITTAL = 2


class QHull:
    def __init__(self):
        self.point_cloud = []
        self.hull = []

    def set_point_cloud(self, volume, volume_size, idx, view):
        """Extract 2D point cloud from a slice of the 3D volume.
        Vectorized with np.argwhere.

        Args:
            volume: 3D numpy array (depth, height, width), uint16
            volume_size: (width, height, depth, channels) tuple
            idx: slice index
            view: CTview enum value
        """
        self.point_cloud = []

        if view == CTview.AXIAL:
            coords = np.argwhere(volume[idx] > 0)  # (N, 2) = (y, x)
            self.point_cloud = [(float(c[1]), float(c[0])) for c in coords]
        elif view == CTview.SAGITTAL:
            coords = np.argwhere(volume[:, :, idx] > 0)  # (N, 2) = (z, y)
            self.point_cloud = [(float(c[0]), float(c[1])) for c in coords]
        elif view == CTview.CORONAL:
            coords = np.argwhere(volume[:, idx, :] > 0)  # (N, 2) = (z, x)
            self.point_cloud = [(float(c[1]), float(c[0])) for c in coords]

    def get_point_cloud(self):
        return self.point_cloud

    def get_drawable_points(self):
        return self.hull

    def initialize(self):
        self.hull = []
        self._jarvis_hull()

    def _jarvis_hull(self):
        """Jarvis March (Gift Wrapping) algorithm for convex hull.
        Exact port of QHull::jarvisHull() from QHull.cpp:172-217
        """
        n = len(self.point_cloud)
        a = self.point_cloud
        if n < 3:
            print("Convex hull not possible")
            return

        # Find the leftmost point
        l = 0
        for i in range(1, n):
            if a[i][0] < a[l][0]:
                l = i

        # Start from leftmost point, keep moving counterclockwise
        p = l
        while True:
            self.hull.append(a[p])

            q = (p + 1) % n
            for i in range(n):
                if self._orientation(a[p], a[i], a[q]) == 2:
                    q = i

            p = q
            if p == l:
                break

    @staticmethod
    def _orientation(p, q, r):
        """Compute orientation of triplet (p, q, r).
        Returns: 0=collinear, 1=clockwise, 2=counterclockwise
        Exact port from QHull.cpp:163-170
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # collinear
        return 1 if val > 0 else 2  # clock or counterclock wise
