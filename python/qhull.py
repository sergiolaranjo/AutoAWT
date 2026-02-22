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
            volume_size: (width, height, depth, channels) tuple (unused, kept for API compat)
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
            if n > 0:
                self.hull = list(a)
            return

        # Find the leftmost point (break ties with lowest y)
        l = 0
        for i in range(1, n):
            if a[i][0] < a[l][0] or (a[i][0] == a[l][0] and a[i][1] < a[l][1]):
                l = i

        # Start from leftmost point, keep moving counterclockwise
        p = l
        while True:
            self.hull.append(a[p])

            q = (p + 1) % n
            for i in range(n):
                ori = self._orientation(a[p], a[i], a[q])
                if ori == 2:
                    # i is more counterclockwise than q
                    q = i
                elif ori == 0 and i != p:
                    # Collinear: pick the farthest point for correct hull
                    dist_i = (a[i][0] - a[p][0])**2 + (a[i][1] - a[p][1])**2
                    dist_q = (a[q][0] - a[p][0])**2 + (a[q][1] - a[p][1])**2
                    if dist_i > dist_q:
                        q = i

            p = q
            if p == l:
                break

    _EPSILON = 1e-10

    @staticmethod
    def _orientation(p, q, r):
        """Compute orientation of triplet (p, q, r).
        Returns: 0=collinear, 1=clockwise, 2=counterclockwise
        Exact port from QHull.cpp:163-170
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < QHull._EPSILON:
            return 0  # collinear
        return 1 if val > 0 else 2  # clock or counterclock wise
