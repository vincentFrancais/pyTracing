""" vector.py
Vector (2-dim and 3-dim, float64) manipulation, operation and factory methods.
Basically convenient functions encapsulating np.ndarray with the corresponding shape and dtype.
Decorated with numba jit.
"""

from typing import TypeVar

import numba
import numpy as np

from .typing import Vec3f, Vec2f

_T = TypeVar("_T", int, float, np.floating, np.integer)
_Tf = TypeVar("_Tf", float, np.floating)

_F_EPS = np.finfo(float).eps


@numba.njit()
def vec2f(x: _Tf = 0., y: _Tf = 0.) -> Vec2f:
    """ Factoru method to create a 2-dim vector (np.ndarray)

    :param x: x value, default is 0
    :type x: number
    :param y: y value, default is 0
    :type y: number
    :return: The vector
    :rtype: Vec2f
    """
    return np.array((x, y), dtype=np.float64)


@numba.njit()
def vec3f(x: _Tf = 0., y: _Tf = 0., z: _Tf = 0.) -> Vec3f:
    """ Factoru method to create a 3-dim vector (np.ndarray)

    :param x: x value
    :type x: number
    :param y: y value
    :type y: number
    :param z: z value
    :type z: number
    :return: Vector
    :rtype: Vec3f
    """
    return np.array((x, y, z), dtype=np.float64)


@numba.njit(fastmath=True)
def normalize(vector: np.ndarray) -> np.ndarray:
    """ Return the vector normalized as V / ||V||

    :param vector: Vector to be normalized
    :type vector: Vec2f or Vec3f
    :return: Normalized Vector
    :rtype: np.ndarray
    """
    return vector / (np.sqrt(np.linalg.norm(vector)) + _F_EPS)


@numba.njit(fastmath=True)
def mix(a: np.ndarray, b: np.ndarray, mix_value: _Tf) -> np.ndarray:
    """Mix two vector such as:
        a * (1 - mix_value) + b * mix_value

    :param a: first vector to mix
    :type a: np.ndarray
    :param b: second vector to mix
    :type b: np.ndarray
    :param mix_value: mixing value
    :type mix_value: float | int
    :return: mixed vector
    :rtype: np.ndarray
    """
    return a * (1. - mix_value) + b * mix_value


@numba.njit(fastmath=True)
def mult_vec_matrix(vector: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """ point-matrix dot product. Point in space is treated as a vector, but with the translation part of the 4x4 mat
    accounted for.
    We suppose that the w components of vector is 1.
    We append 1. at the end of vector: v = [x, y, z, 1]
    Then we compute the dot product with the matrix and extract the w component
    Then we return the vec3 divided by w
    """
    tmp = np.append(vector, 1.).dot(mat)
    res = tmp[:-1]
    w = tmp[-1]
    return res / w


@numba.njit(fastmath=True)
def mult_dir_matrix(vector: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Vector-Matrix dot product. Here we consider a Vector (from a mathematical point of view).
    So the translation component of the 4x4 matrix is irrelevant.
    """
    return vector.dot(mat[:-1, :-1])
