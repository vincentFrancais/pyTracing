from pytracing import Renderer
from pytracing import Scene

if __name__ == "__main__":
    scene = Scene()
    scene.add_spheres_random(2)

    renderer = Renderer(width=300, height=200, render_type="py")
    renderer.render(scene)
