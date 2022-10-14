
# pyTracing

A very basic ray tracer in plain python (numpy) and numba jit. At the moment only implicit spheres are supported by the ray tracer.

The fast renderer use jit compilation for fast ray casting. It is capable to render an 1024x768 images containing 10 spheres with reflections and shading in about 10 seconds.

The pure python use numpy to hold and compute most vector algebra. It is much slower but can resolve a 640x480 image with 10 spheres in 60 seconds. It does not support reflections nor diffuse and specular shading.

Needs Python >=3.10. Should work on python 3.9 but not tested.

## Usage/Examples

```python
import pytracing

scene = pytracing.Scene()       # Declare our scene object
scene.add_spheres_random(10)    # Add ten randomized spheres to our scene

# Decalre our renderer
# This will render an image of 640x480 pixels using the jit ray tracer and write the image to my_render.png
renderer = pytracing.Renderer(width=640, height=480, render_type="fast", out_file="my_render.png")
renderer.render(scene)  # Render !
```

## References
https://www.scratchapixel.com/

https://github.com/rafael-fuente/Python-Raytracer

https://excamera.com/sphinx/article-ray.html

https://medium.com/swlh/ray-tracing-from-scratch-in-python-41670e6a96f9

https://gist.github.com/rossant/6046463