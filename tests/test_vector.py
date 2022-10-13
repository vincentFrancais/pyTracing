import numpy as np

from pytracing.geometry.vector import Vec3f


def test_vec_initialisations():
    # Test the init methods for vectors
    v = Vec3f(x=1, y=2, z=3)
    v2 = np.asarray([1, 2, 3], dtype=np.float32)
    for i in range(3):
        assert v[i] == v2[i]

def test_vec3_math():
    # test vector comparisons
    v1 = Vec3f(x=1, y=2, z=3)
    v11 = Vec3f(x=1, y=2, z=3)
    assert v1 == v11
    assert v1 != 1
    assert v1 != "test"

    # test Vector add
    v2 = Vec3f(x=2, y=3, z=4)
    assert v1 + v2 == Vec3f(3, 5, 7)
    v1 += v2
    assert v1 == Vec3f(3, 5, 7)
    for val in (1, 1.):
        v1 = Vec3f(x=1, y=2, z=3)
        assert v1 + val == Vec3f(2, 3, 4)
        v1 += val
        assert v1 == Vec3f(2, 3, 4)

    # test vector sub
    v1 = Vec3f(x=1, y=2, z=3)
    assert v1 - v2 == Vec3f(-1, -1, -1)
    v1 -= v2
    assert v1 == Vec3f(-1, -1, -1)
    for val in (1, 1.):
        v1 = Vec3f(x=1, y=2, z=3)
        assert v1 - val == Vec3f(0, 1, 2)
        v1 -= val
        assert v1 == Vec3f(0, 1, 2)

    # test mul
    # test div
    # test matmul

def test_vec3_normalize():
    v = Vec3f(3, 4, 6)
    v_copy = Vec3f(3, 4, 6)
    norm = v.norm()
    assert norm == 3*3 + 4*4 + 6*6
    v.normalize()
    assert v == v_copy / np.sqrt(norm)