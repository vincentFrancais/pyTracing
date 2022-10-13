import numpy as np

from pytracing.geometry.vector import Vec3f
from pytracing.shapes.shape import Sphere, ImplicitObject


def test_ray_hit():
    ray_orig = Vec3f(0, 0, 0)
    ray_dir = Vec3f(0, 0, 1)
    ray_dir.normalize()
    sphere = Sphere(center=Vec3f(0, 0, 3), radius=1)
    print(ray_orig, ray_dir, sphere)
    t = sphere.intersect(ray_orig, ray_dir)
    assert t > 0 and t != np.inf


def test_ray_miss():
    ray_orig = Vec3f(0, 0, 0)
    ray_dir = Vec3f(0, 1, 0)
    ray_dir.normalize()
    sphere = Sphere(center=Vec3f(0, 0, 3), radius=1)
    print(ray_orig, ray_dir, sphere)
    t = sphere.intersect(ray_orig, ray_dir)
    assert t == np.inf