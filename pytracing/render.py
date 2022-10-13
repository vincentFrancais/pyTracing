import collections
import time
from typing import Iterable

import numba
import numpy as np
import tqdm
from PIL import Image

from pytracing.geometry import Vec3f
from pytracing.shapes.shape import Sphere, ImplicitObject

Options = collections.namedtuple("Options", ["width", "height", "fov", "camera_to_world"])


def trace(ray_origin: Vec3f, ray_dir: Vec3f, objs: Iterable[ImplicitObject]):
    t_near = np.inf
    hit_object = None
    for o in objs:
        t = o.intersect(ray_orig=ray_origin, ray_dir=ray_dir)
        if t < t_near:
            hit_object = o
            t_near = t

    return t_near, hit_object


def cast_ray(ray_origin: Vec3f, ray_dir: Vec3f, objs):
    """Cast a ray on the scene and get the corresponding color of the pixel it is originating from

    :param ray_origin:
    :param ray_dir:
    :param objs:
    :return:
    """
    hit_color = Vec3f(0, 0, 0)
    scale = 4

    # we first compute the intersection for each object in the scene
    ts = np.asarray([o.intersect(ray_origin, ray_dir) for o in objs], dtype=np.float64)

    # Get the index of the nearest hit
    # and get the nearest distance
    nearest_idx = np.argmin(ts)
    t = ts[nearest_idx]
    if t == np.inf:
        # if the nearest distance is still np.inf, it means the ray has not hit anything.
        # so we return a black pixel
        return hit_color

    # Get the hit position and object
    hit_object = objs[nearest_idx]
    p_hit = ray_origin + ray_dir * t

    # the normal vector and the texture data are retrieved from the hit object
    normal_hit, tex = hit_object.get_surface_data(p_hit)

    # Checkerboard pattern texturing
    pattern = (np.fmod(tex[0] * scale, 1) > 0.5) ^ (np.fmod(tex[1] * scale, 1) > 0.5)

    # Mix the object color with the checkerboard pattern
    mixx = hit_object.color * (1. - pattern) + hit_object.color * 0.8 * pattern

    # Get the actual color of the pixel from, depending on where he ray has hit the object
    hit_color = np.maximum(0., normal_hit.dot(-ray_dir)) * mixx
    return hit_color


# def cast_ray(ray_origin: Vec3f, ray_dir: Vec3f, objs):
#     hit_color = Vec3f(0, 0, 0)
#     scale = 4
#     t, hit_object = trace(ray_origin, ray_dir, objs)
#     if hit_object:
#         p_hit = ray_origin + ray_dir * t
#         normal_hit, tex = hit_object.get_surface_data(p_hit)
#         pattern = (np.fmod(tex.x * scale, 1) > 0.5) ^ (np.fmod(tex.y * scale, 1) > 0.5)
#         hit_color = np.maximum(0., normal_hit.dot(-ray_dir)) * Vec3f.mix(hit_object.color,
#                                                                          hit_object.color * 0.8, pattern)
#     return hit_color


# def ray_trace(x, y, opts, objs, ray_origin):
#     # print(x)
#     # xs, ys = grid
#     print(x)
#     ray_dir = Vec3f(x, y, -1)
#     # ray_dir.mult_dir_matrix(opts.camera_to_world)
#     ray_dir = ray_dir.norm()
#     color = cast_ray(ray_origin, ray_dir, objs)


def compute_frame_buffer(opts: Options, objs):
    frame_buffer = np.empty((opts.width * opts.height, 3), dtype=np.float32)
    scale = np.tan(np.deg2rad(opts.fov * 0.5))
    image_aspect_ratio = opts.width / float(opts.height)
    ray_origin = Vec3f(0, 0, 0)
    ray_origin.mult_vect_matrix(opts.camera_to_world)
    c = 0
    bar = tqdm.tqdm(total=opts.width*opts.height)
    for j in range(opts.height):
        for i in range(opts.width):
            x = (2 * (i + 0.5) / opts.width - 1) * image_aspect_ratio * scale
            y = (1 - 2 * (j + 0.5) / opts.height) * scale
            ray_dir = Vec3f(x, y, -1)
            ray_dir.mult_dir_matrix(opts.camera_to_world)
            ray_dir.normalize()
            color = cast_ray(ray_origin, ray_dir, objs)
            frame_buffer[c] = color
            c += 1
            bar.update()
    return frame_buffer


def make_png(fb: np.ndarray, opts):
    # The frame buffer is first clamped to [0-255]
    fb = fb.clip(0, 1) * 255
    # extract the R G B channels (first, second and third col of the frame buffer)
    # each channel is reshaped to the actual image size and converted to uint8
    # Each channel is converted to grayscale PIL Image, then merged to a RGB Image
    rgb = [Image.fromarray(fb[..., i].reshape((opts.height, opts.width)).astype(np.uint8), "L") for i in range(3)]
    Image.merge("RGB", rgb).save("pure-python.png")


def make_ppm(opts: Options, frame_buffer):
    with open("./out.ppm", "w") as f:
        f.write(f"P3\n{opts.width} {opts.height}\n255\n")
        for v in frame_buffer:
            r = 255 * np.clip(v.x, 0, 1)
            g = 255 * np.clip(v.y, 0, 1)
            b = 255 * np.clip(v.z, 0, 1)
            f.write(f"{int(r)} {int(g)} {int(b)}\n")


def render(opts: Options, objs):
    # compute_frame_buffer(opts, objs)
    tic = time.perf_counter()
    frame_buffer = compute_frame_buffer(opts, objs)
    toc = time.perf_counter()
    print(f"Frame buffer computed in {toc-tic:0.4f} sec")
    print("writing ppm")
    # make_ppm(opts, frame_buffer)
    make_png(frame_buffer, opts)


