from collections import namedtuple

import numba
import numpy as np

from ..geometry.typing import Vec3f


@numba.njit()
def intersect_sphere(ray_orig: Vec3f, ray_dir: Vec3f, center: Vec3f, radius: float) -> float:
    """Given a sphere with center C and radius R, a ray with origin O and direction D, and a point in space
    P = O + tD, we have the parametric equation:
        |P - C|^2 - R^2 = 0
        |O + tD - C|^2 - R^2 = 0

    Developing this we obtain the quadratic equation:
        a*t^2 + b*t + c = 0
    with:
        a = D^2 = 1
        b = 2D(O-C)
        c = |O-C|^2 - R^2
    Thus the roots determine if the ray has hit the sphere, and if the ray is originating from inside or behind it.

    :param orig:
    :param dir:
    :return:
    """
    radius2 = radius * radius
    # print("intersect: orig=", orig, "dir=", dir)
    center_dist = ray_orig - center
    # print(ray_dir)
    # print("center_dist", center_dist)
    a = np.dot(ray_dir, ray_dir)
    # print("a", a)
    b = np.dot(ray_dir, center_dist) * 2.
    # print("b", b)
    c = center_dist.dot(center_dist) - radius2
    # print("c", c)
    discr = b * b - 4. * a * c

    # Rewriting the roots to be more computational efficient (Scratchpixel)
    q = -0.5 * (b + np.sign(b) * np.sqrt(np.maximum(0, discr)))  # We don't want neg delta
    t0 = q / a
    t1 = c / q

    # Select the actual distance, which is the lowest root
    t = np.where((t0 > 0) & (t0 < t1), t0, t1).item()

    # if the discriminant or the roots are negative
    # we did not hit anything, so we return np.inf
    hit = (discr > 0) & (t > 0)
    return np.where(hit, t, np.inf).item()


Sphere = namedtuple("Sphere", ("center", "radius", "color", "intersect"))


def add_sphere(center: Vec3f, radius: float, color: Vec3f):
    return Sphere(center, radius, color, intersect_sphere)
