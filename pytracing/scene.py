from __future__ import annotations

import random
from collections import namedtuple
from typing import Iterable, Optional

_SphereDescript = namedtuple("_SphereDescript", ["type", "center", "radius", "color", "is_checkerboard"])


# @dataclasses.dataclass
# class SceneObjectDescriptor:
#     type: str
#     center:


class Scene:
    def __init__(self):
        """Class holdings the scene objects.
        At the moment only spheres are implemented
        """
        self._objects = []

    def add_object(self):
        pass

    def add_sphere(self, center: Iterable[float | int], radius: float | int,
                   color: Optional[Iterable[float | int]] = None, is_checkerboard: bool = False):
        """ Add a sphere to the scene

        :param center: Center of sphere
        :type center: Iterable[float | int]
        :param radius: Sphere radius
        :type radius: float
        :param color: Sphere diffuse color. If None a random color is assigned
        :type color: Iterable[float | int] or None
        :param is_checkerboard:
        :type is_checkerboard:
        """
        sphere = _SphereDescript(center=center, radius=radius, color=color,
                                 type="sphere", is_checkerboard=is_checkerboard)
        self._objects.append(sphere)

    def add_spheres_random(self, n_spheres: int):
        """ Add random spheres to the scene.
        The spheres has a position in [-5, 5], and a radius in [0.5, 1]

        :param n_spheres: Number of spheres to randomly add
        :type n_spheres: int
        """
        for i in range(n_spheres):
            x = (0.5 - random.random()) * 10.
            y = (0.5 - random.random()) * 10.
            z = (0.5 + random.random()) * 10.
            radius = 0.5 * random.random() + 0.5
            color = (random.random(), random.random(), random.random())
            is_checkerboard = random.random() > 0.5
            self.add_sphere(center=(x, y, z), radius=radius, color=color, is_checkerboard=is_checkerboard)

    @property
    def objects(self):
        """ The object list
        """
        return self._objects

    def __len__(self):
        return len(self._objects)
