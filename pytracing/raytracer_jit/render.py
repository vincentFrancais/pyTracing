import collections
import time
from typing import NewType

import numba
import numpy as np
from numba import literal_unroll
from numba.typed import List as NumbaList

from pytracing.raytracer_jit.geometry.vector import vec3f, normalize
from pytracing.raytracer_jit.shapes.shape import Sphere
from pytracing.scene import Scene

Vec3f = NewType("Vec3f", np.ndarray)

Options = collections.namedtuple("Options", ["width", "height", "fov", "camera_to_world", "max_bounces"])

L = vec3f(10, 20, -5)  # Light
L_color = vec3f(1, 1, 1)


@numba.jit(nopython=True, fastmath=True)
def shade_plane(hit_point: Vec3f, object_color: Vec3f):
    ix = int(np.floor(hit_point[0])) % 2
    iy = int(np.floor(hit_point[1]))
    iz = int(np.floor(hit_point[2])) % 2
    color_1 = object_color
    color_2 = vec3f(0, 0, 0)
    if (ix and iy) or (ix == 0 and iz == 0):
        return color_1
    return color_2


@numba.jit(nopython=False, fastmath=True)
def cast_ray_v2(ray_origin: Vec3f, ray_dir: Vec3f, objs):
    hit_color = np.asarray((0, 0, 0), dtype=np.float64)
    # ts = np.empty((len(objs)), dtype=np.float64)
    # for o in literal_unroll(objs):
    #     ts[]
    # ts = []
    # for o in literal_unroll(objs):
    #     t = o.intersect(ray_origin, ray_dir)
    #     print(t)
    #     ts.append(t)
    # ts = np.asarray(ts, dtype=np.float64)
    ts = np.asarray([o.intersect(ray_origin, ray_dir) for o in literal_unroll(objs)], dtype=np.float64)
    # Get the index of the nearest hit
    # and get the nearest distance
    nearest_idx = np.argmin(ts)
    t = ts[nearest_idx]
    if t == np.inf:
        # if the nearest distance is still np.inf, it means the ray has not hit anything.
        # so we return a black pixel
        return hit_color
    hit_object = objs[nearest_idx]
    # hit_object = get_object(objs, nearest_idx)
    p_hit = ray_origin + ray_dir * t
    # print(p_hit)
    hit_color = hit_object.shade(p_hit, ray_dir)
    return hit_color


@numba.jit(nopython=True, fastmath=True)
def cast_ray(ray_origin: Vec3f, ray_dir: Vec3f, bounce: int, max_bounces: int, objs):
    """Cast a ray on the scene.
    If an object is hit by the ray, compute the diffuse and specular shading along with the reflection.
    The reflection is computed by casting rays from the previous ray hit coordinates.

    :param ray_origin: ray origin coordinates
    :param ray_dir: ray direction normalized
    :param bounce: if > 1, this is a reflection ray.
    :param max_bounces: Maximum secondary rays ti generate
    :param objs: list of objects in the scene
    :return:
    """
    background = np.asarray((105. / 255., 105. / 255., 105. / 255.), dtype=np.float64)

    # we first compute the intersection for each object in the scene
    ts = np.asarray([o.intersect(ray_origin, ray_dir) for o in objs], dtype=np.float64)

    # Get the index of the nearest hit
    # and get the nearest distance
    nearest_idx = np.argmin(ts)
    t = ts[nearest_idx]
    if t == np.inf:
        # if the nearest distance is still np.inf, it means the ray has not hit anything.
        # so we return a black pixel
        return background

    # Get the hit position and object
    hit_object = objs[nearest_idx]
    p_hit = ray_origin + ray_dir * t

    col_ray = vec3f(0.05, 0.05, 0.05)
    normal = hit_object.get_normal(p_hit)
    to_light = normalize(L - p_hit)
    to_origin = normalize(ray_origin - p_hit)
    light_distances = [o.intersect(p_hit + normal * 0.0001, to_light) for idx, o in enumerate(objs) if
                       idx != nearest_idx]
    light_distances = np.asarray(light_distances, dtype=np.float64)
    if len(light_distances) > 0:
        # We first check that there is other objects in the scene
        if np.min(light_distances) < np.inf:
            # We are shadowed !!
            col_ray *= 0

    # Lambert shading (diffuse)
    col_ray += hit_object.lambert_shading(p_hit, to_light)

    if bounce < max_bounces:
        # We actually want to see a result before next year, so we limit the bounces
        new_ray_dir = ray_dir - normal * 2 * ray_dir.dot(normal)
        new_ray_dir = normalize(new_ray_dir)
        new_ray_origin = p_hit + normal * 0.0001
        col_ray += cast_ray(new_ray_origin, new_ray_dir, bounce + 1, max_bounces, objs) * 0.75

    # TODO: does not work well
    # Blinn-Phong shading (specular)
    # phong = normal.dot(normalize(to_light + to_origin))
    # phong = np.minimum(1, np.maximum(phong, 0))  # clip
    # col_ray += vec3f(1, 1, 1) * np.power(phong, 50)
    return col_ray


@numba.jit(nopython=True, fastmath=True)
def compute_frame_buffer(opts: Options, objs):
    """ Compute the frame buffer by casting a ray per pixel.
    """
    frame_buffer = np.empty((opts.width * opts.height, 3), dtype=np.float64)
    scale = np.tan(np.deg2rad(opts.fov * 0.5))
    image_aspect_ratio = opts.width / float(opts.height)

    # Get the ray origin (0, 0, 0) and project it to world coords
    tmp = np.asarray((0, 0, 0), dtype=np.float64)
    transformed = np.append(tmp, 1.).dot(opts.camera_to_world)
    ray_origin = transformed[:-1] / transformed[-1]
    c = 0
    for j in range(opts.height):
        for i in range(opts.width):
            # ray direction
            # Compute the x, position in screen space (from the camera space)
            # We want the ray to traverse the center of the pixel and square pixels
            x = (2 * (i + 0.5) / opts.width - 1) * image_aspect_ratio * scale
            y = (1 - 2 * (j + 0.5) / opts.height) * scale
            # print(x, y)
            # ray direction in camera coords to world coords, normalized
            ray_dir = np.asarray((x, y, -1), dtype=np.float64).dot(opts.camera_to_world[:-1, :-1])
            ray_dir /= np.sqrt(np.linalg.norm(ray_dir))  # normalized
            color = cast_ray(ray_origin, ray_dir, 0, opts.max_bounces, objs)
            frame_buffer[c] = color
            c += 1
    return frame_buffer


def make_ppm(opts: Options, frame_buffer):
    with open("./out.ppm", "w") as f:
        f.write(f"P3\n{opts.width} {opts.height}\n255\n")
        for v in frame_buffer:
            r = 255 * np.clip(v[0], 0, 1)
            g = 255 * np.clip(v[1], 0, 1)
            b = 255 * np.clip(v[2], 0, 1)
            f.write(f"{int(r)} {int(g)} {int(b)}\n")


def _convert_scene_to_numba_list(scene: Scene):
    objects_list = NumbaList()
    for o in scene.objects:
        objects_list.append(o)
    return objects_list


def render(opts: Options, scene: Scene):
    objs = scene_factory(scene)
    tic = time.perf_counter()
    fb = compute_frame_buffer(opts, objs)
    toc = time.perf_counter()
    print(f"Frame buffer computed in {toc - tic:0.4f} sec")
    return fb


def scene_factory(scene: Scene):
    """Convert a Scene object to a list of Numba jitclasses objects
    Only sphere are supported.
    """
    objects_list = NumbaList()
    for o in scene.objects:
        if o.type != "sphere":
            raise ValueError("Fast renderer supports only sphere objects")
        center = vec3f(*o.center)
        color = vec3f(*o.color)
        o_jit = Sphere(center, o.radius, color, checkerboard=o.is_checkerboard)
        objects_list.append(o_jit)
    return objects_list
