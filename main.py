import time

import numpy as np

from pytracing.geometry import Vec3f
from pytracing.render import render, Options
from pytracing.shapes.shape import Sphere


def main():
    camera_to_world = np.array([(0.945519, 0, -0.325569, 0),
                                (-0.179534, 0.834209, -0.521403, 0),
                                (0.271593, 0.551447, 0.78876, 0),
                                (4.208271, 8.374532, 17.932925, 1)], dtype=np.dtype(np.float32))

    options = Options(width=640, height=480, fov=51.52, camera_to_world=camera_to_world)

    n_spheres = 3
    objects = []
    pp = [Vec3f(4.6961427, -3.37284, 7.0039153),
          Vec3f(0.7757796, 4.8313227, 5.773529),
          Vec3f(03.421212, 2.6662118, 14.42348),
          Vec3f(-3.0861077, 3.2347806, 8.906423),
          Vec3f(0.74072623, -2.7847292, 10.152576)]
    rr = [0.9931678485059355,
          0.7339902015024757,
          0.8478040771121532,
          0.5060574577048714,
          0.5277527993544071, ]
    for ii in range(n_spheres):
        # x = (0.5 - random.random()) * 10.
        # y = (0.5 - random.random()) * 10.
        # z = (0.5 + random.random()) * 10.
        # pos = Vec3f(x, y, z)
        # radius = 0.5 * random.random() + 0.5
        objects.append(Sphere(pp[ii], rr[ii]))

    for o in objects:
        print(o)

    render(options, objects)


if __name__ == "__main__":
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print(f"Render time {toc - tic:0.4f} sec")
