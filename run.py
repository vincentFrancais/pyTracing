from pytracing import Renderer
from pytracing import Scene

if __name__ == "__main__":
    scene = Scene()
    scene.add_spheres_random(10)

    renderer = Renderer(width=1024, height=768, render_type="fast")
    renderer.render(scene)
