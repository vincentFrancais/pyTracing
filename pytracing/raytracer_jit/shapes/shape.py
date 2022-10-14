import random
from typing import Tuple, NewType

import numba
import numpy as np
from numba.experimental import jitclass

from ..geometry.vector import vec3f, vec2f

Vec3f = NewType("Vec3f", np.ndarray)
Vec2f = NewType("Vec2f", np.ndarray)

_F_EPS = np.finfo(float).eps

sphere_spec = [
    ("color", numba.float64[:]),
    ("center", numba.float64[:]),
    ("radius", numba.float64),
    ("radius2", numba.float64),
    ("checkerboard", numba.boolean)
]


@jitclass(sphere_spec)
class Sphere:
    def __init__(self, center: Vec3f, radius: float, color: Vec3f = vec3f(0.5, 0.5, 0.5), checkerboard: bool = False):
        """Create a sphere primitive

        :param center: Center of sphere
        :type center: Vec3f
        :param radius: Sphere radius
        :type radius: float
        :param checkerboard: Should render this sphere with a checkerboard pattern texture
        :type checkerboard: bool
        """
        self.color = color
        self.center = center
        self.radius = radius
        self.radius2 = radius * radius
        self.checkerboard = checkerboard

    def __str__(self):
        return f"Sphere(C={self.center}, R={self.radius})"

    def get_surface_data(self, point_hit: Vec3f):
        """Return the normal to hit and UV

        :param point_hit: ray hit coordinates
        :type point_hit: Vec3f
        :return: normal to hit point and UV
        :rtype: Tuple[Vec3f, Vec2f]
        """
        normal = point_hit - self.center
        normal = normal / np.sqrt(np.linalg.norm(normal) + _F_EPS)
        tex = np.asarray((0, 0), dtype=np.float64)
        tex[0] = (1. + np.arctan2(normal[2], normal[0]) / np.pi) * 0.5
        tex[1] = np.arccos(normal[1]) / np.pi
        return normal, tex

    def get_normal(self, point_hit: Vec3f):
        """Return the normal to hit point

        :param point_hit: ray hit coordinates
        :type point_hit: Vec3f
        :return: Normal
        :rtype: Vec3f
        """
        normal = point_hit - self.center
        normal = normal / np.sqrt(np.linalg.norm(normal) + _F_EPS)
        return normal

    def lambert_shading(self, p_hit: Vec3f, to_light: Vec3f) -> Vec3f:
        """ Compute the Lambert shading (diffuse shading)

        :param p_hit: hit point
        :type p_hit: Vec3f
        :param to_light: Direction from hit point to light
        :type to_light: Vec3f
        :return: The diffuse color component
        :rtype: Vec3f
        """
        # color = vec3f(0.05, 0.05, 0.05)
        diffuse = self.diffuse_color(p_hit)
        lv = np.maximum(self.get_normal(p_hit).dot(to_light), 0)
        diffuse_color = diffuse * lv
        return diffuse_color

    def diffuse_color(self, p_hit: Vec3f):
        """ Get the diffuse color texture of the sphere. If `checkerboard` is True, the diffuse color is blended
        with a checkerboard texture

        :param p_hit: hit point
        :type p_hit: Vec3f
        :return: Diffure color
        :rtype: Vec3f
        """
        if not self.checkerboard:
            return self.color
        normal_hit, tex = self.get_surface_data(p_hit)
        scale = 4
        # Checkerboard pattern texturing
        pattern = (np.fmod(tex[0] * scale, 1) > 0.5) ^ (np.fmod(tex[1] * scale, 1) > 0.5)

        # Mix the object color with the checkerboard pattern
        mixx = self.color * (1. - pattern) + self.color * 0.8 * pattern
        return mixx

    def shade(self, p_hit: Vec3f, ray_dir: Vec3f):
        # the normal vector and the texture data are retrieved from the hit object
        normal_hit, tex = self.get_surface_data(p_hit)
        scale = 4
        # Checkerboard pattern texturing
        pattern = (np.fmod(tex[0] * scale, 1) > 0.5) ^ (np.fmod(tex[1] * scale, 1) > 0.5)

        # Mix the object color with the checkerboard pattern
        mixx = self.color * (1. - pattern) + self.color * 0.8 * pattern

        # Get the actual color of the pixel from, depending on where he ray has hit the object
        hit_color = np.maximum(0., normal_hit.dot(-ray_dir)) * mixx
        return hit_color

    def intersect(self, ray_orig: Vec3f, ray_dir: Vec3f) -> float:
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

        :param ray_orig: Ray origin point
        :type ray_orig: Vec3f
        :param ray_dir: Ray direction (normalized)
        :return: The t parameter, such as in P = O + tD if P is the hit point. If there is no hit, return np.inf
        """
        # print("intersect: orig=", orig, "dir=", dir)
        center_dist = ray_orig - self.center
        # print(ray_dir)
        # print("center_dist", center_dist)
        a = np.dot(ray_dir, ray_dir)
        # print("a", a)
        b = np.dot(ray_dir, center_dist) * 2.
        # print("b", b)
        c = center_dist.dot(center_dist) - self.radius2
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


plane_spec = [
    ("color", numba.float64[:]),
    ("center", numba.float64[:]),
    ("normal", numba.float64[:]),
]


@jitclass(plane_spec)
class InfinitePlane:
    def __init__(self, pos: Vec3f, normal: Vec3f):
        self.color = np.asarray([random.random(), random.random(), random.random()], dtype=np.float64)
        self.center = pos
        self.normal = normal

    def intersect(self, ray_orig: Vec3f, ray_dir: Vec3f) -> float:
        denom = np.dot(self.normal, ray_dir)
        if abs(denom) < 1.e-6:
            return np.inf
        t = np.dot(self.center - ray_orig, self.normal) / denom
        # return np.where(t>0, t, np.inf)
        return t if t > 0 else np.inf

    def get_surface_data(self, point_hit: Vec3f):
        return self.normal, vec2f(0, 0)

    def shade(self, p_hit: Vec3f, ray_dir: Vec3f):
        ix = int(np.floor(p_hit[0])) % 2
        iy = int(np.floor(p_hit[1]))
        iz = int(np.floor(p_hit[2])) % 2
        color_1 = self.color
        color_2 = vec3f(0, 0, 0)
        if (ix and iy) or (ix == 0 and iz == 0):
            return color_1
        return color_2
