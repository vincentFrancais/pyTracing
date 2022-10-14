import collections
from typing import Literal

import numpy as np
from PIL import Image

from .raytracer_jit.render import render as render_jit
from .raytracer_py.render import render as render_py

from .scene import Scene

Options = collections.namedtuple("Options", ["width", "height", "fov", "camera_to_world", "max_bounces"])


def make_default_camera(precision):
    camera_to_world = np.array([(0.945519, 0, -0.325569, 0),
                                (-0.179534, 0.834209, -0.521403, 0),
                                (0.271593, 0.551447, 0.78876, 0),
                                (4.208271, 8.374532, 17.932925, 1)], dtype=precision)
    return camera_to_world


class Renderer:
    def __init__(self, width: int = 640, height: int = 480,
                 render_type: Literal["fast", "py"] = "fast"):
        self._width = width
        self._height = height
        self._fov = 51.52
        self._render_type = render_type
        # self._precision = np.float64 if self._render_type == "fast" else np.float32

    def render(self, scene: Scene, out_file: str = "render.png"):
        if self._render_type == "fast":
            frame_buffer = self._render_jit(scene)
        elif self._render_type == "py":
            frame_buffer = self._render_py(scene)
        else:
            raise ValueError(f"Unknown render type {self._render_type}")
        self.make_png(frame_buffer, out_file)

    def _render_py(self, scene: Scene):
        max_bounces = 3
        precision = np.float32
        render_options = Options(self._width, self._height, self._fov,
                                 make_default_camera(precision), max_bounces)
        return render_py(render_options, scene)

    def _render_jit(self, scene: Scene):
        max_bounces = 3
        precision = np.float64
        render_options = Options(self._width, self._height, self._fov,
                                 make_default_camera(precision), max_bounces)
        return render_jit(render_options, scene)

    def make_png(self, frame_buffer: np.ndarray, out_file: str):
        # The frame buffer is first clamped to [0-255]
        fb = frame_buffer.clip(0, 1) * 255
        # extract the R G B channels (first, second and third col of the frame buffer)
        # each channel is reshaped to the actual image size and converted to uint8
        # Each channel is converted to grayscale PIL Image, then merged to a RGB Image
        rgb = [Image.fromarray(fb[..., i].reshape((self._height, self._width)).astype(np.uint8), "L") for i in range(3)]
        Image.merge("RGB", rgb).save(out_file)
