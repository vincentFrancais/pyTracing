import random
from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

from ..geometry.vector import Vec3f, Vec2f


_F_EPS = np.finfo(float).eps
FARAWAY = 1.0e39


class ImplicitObject(metaclass=ABCMeta):
    def __init__(self):
        self.color = Vec3f(random.random(), random.random(), random.random())

    @abstractmethod
    def intersect(self, ray_orig: Vec3f, ray_dir: Vec3f) -> float:
        """Check if a ray has intersected with the object

        :param ray_orig: Ray origin
        :type ray_orig: Vec3f
        :param ray_dir: Ray direction (normalized)
        :type ray_dir: Vec3f
        :return:
        """

    @abstractmethod
    def get_surface_data(self, point_hit: Vec3f) -> Tuple[Vec3f, Vec2f]:
        """

        :param point_hit:
        :return:
        """


class Sphere(ImplicitObject):
    def __init__(self, center: Vec3f, radius: float):
        super(Sphere, self).__init__()
        self.center = center
        self.radius = radius
        self.radius2 = radius * radius

    def __str__(self):
        return f"Sphere(C={self.center}, R={self.radius})"

    def get_surface_data(self, point_hit: Vec3f):
        normal: Vec3f = point_hit - self.center
        normal.normalize()
        tex = Vec2f(0, 0)
        tex.x = (1. + np.arctan2(normal.y, normal.z) / np.pi) * 0.5
        tex.y = np.arccos(normal.x) / np.pi
        return normal, tex

    def intersect(self, ray_orig: Vec3f, ray_dir: Vec3f) -> float:
        """Given a sphere with center C and radius R, a ray with origin O and direction D, and a point in space
        P = O + tD, we have the parametric equation:
            | P - C|^2 - R^2 = 0
            |O + tD - C|^2 - R^2 = 0

        Developing this we obtain the quadratic equation:
            a*t^2 + b*t + c = 0
        with:
            a = D^2 = 1
            b = 2D(O-C)
            c = |O-C|^2 - R^2
        Thus the roots determine if the ray has hit the sphere, and if the ray is originating from inside or behind it.

        :param ray_orig:
        :param ray_dir:
        :return:
        """
        center_dist = ray_orig - self.center
        a = np.dot(ray_dir, ray_dir)
        b = np.dot(ray_dir, center_dist) * 2.
        c = center_dist.dot(center_dist) - self.radius2
        discr = b * b - 4. * a * c
        q = -0.5 * (b + np.sign(b) * np.sqrt(np.maximum(0, discr)))  # We don't want neg delta
        t0 = q / a
        t1 = c / q
        t = np.where((t0 > 0) & (t0 < t1), t0, t1).item()
        hit = (discr > 0) & (t > 0)
        return np.where(hit, t, np.inf).item()

