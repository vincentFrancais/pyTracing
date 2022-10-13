from collections import namedtuple

import numba
import numpy as np

from ..geometry.typing import Vec3f


@numba.njit()
def intersect_plane(ray_origin: Vec3f, ray_dir: Vec3f, plane_center: Vec3f, plane_normal: Vec3f):
    denom = np.dot(plane_normal, ray_dir)
    if abs(denom) < 1.e-6:
        return np.inf
    t = np.dot(plane_center - ray_origin, plane_normal) / denom
    # return np.where(t>0, t, np.inf)
    return t if t > 0 else np.inf


Plane = namedtuple("Plane", ("center", "normal", "color", "intersect"))


def add_plane(center: Vec3f, normal: Vec3f, color: Vec3f):
    return Plane(center, normal, color, intersect_plane)
