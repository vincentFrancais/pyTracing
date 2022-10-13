import random

import numpy as np
from numba.typed import List as NumbaList

from pytracing_numba.geometry.vector import vec3f
from pytracing_numba.render import Options, render
from pytracing_numba.shapes.shape import Sphere


def main():
    camera_to_world = np.array([(0.945519, 0, -0.325569, 0),
                                (-0.179534, 0.834209, -0.521403, 0),
                                (0.271593, 0.551447, 0.78876, 0),
                                (4.208271, 8.374532, 17.932925, 1)], dtype=np.dtype(np.float64))

    options = Options(width=1024, height=768, fov=51.52, camera_to_world=camera_to_world, max_bounces=5)

    n_spheres = 20
    objects = NumbaList()
    for ii in range(n_spheres):
        x = (0.5 - random.random()) * 10.
        y = (0.5 - random.random()) * 10.
        z = (0.5 + random.random()) * 10.
        pos = vec3f(x, y, z)
        radius = 0.5 * random.random() + 0.5
        objects.append(Sphere(pos, radius))
    # objects.append(Sphere(vec3f(0, -100, 0), 40))
    render(options, objects)


if __name__ == "__main__":
    main()
