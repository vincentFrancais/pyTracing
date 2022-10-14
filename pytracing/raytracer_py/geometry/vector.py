from __future__ import annotations

from typing import Optional, TypeVar, Protocol

import numpy as np

T = TypeVar("T", int, float, np.floating, np.integer)
Tf = TypeVar("Tf", float, np.floating)

_F_EPS = np.finfo(float).eps


def vec2f(x: Optional[Tf] = None, y: Optional[Tf] = None) -> np.ndarray:
    vals = (x or 0., y or 0.)
    return np.array(vals, dtype=np.float64)


def vec3f(x: Optional[Tf] = None, y: Optional[Tf] = None, z: Optional[Tf] = None) -> np.ndarray:
    vals = (x or 0., y or 0., z or 0.)
    return np.array(vals, dtype=np.float64)


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / (np.sqrt(np.linalg.norm(vector)) + _F_EPS)


def mix(a: np.ndarray, b: np.ndarray, mix_value: Tf) -> np.ndarray:
    return a * (1. - mix_value) + b * mix_value


def mult_vec_matrix(vector: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """ point-matrix dot product. Point in space is treated as a vector, but with the translation part of the 4x4 mat
    accounted for.
    We suppose that the w components of vector is 1.
    We append 1. at the end of vector: v = [x, y, z, 1]
    Then we compute the dot product with the matrix and extract the w component
    Then we return the vec3 divided by w

    :param vector:
    :param mat:
    :return:
    """
    tmp = np.append(vector, 1.).dot(mat)
    res = tmp[:-1]
    w = tmp[-1]
    return res / w


def mult_dir_matrix(vector: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Vector-Matrix dot product. Here we consider a Vector (from a mathematical point of view).
    So the translation component of the 4x4 matrix is irrelevant.

    :param vector:
    :param mat:
    :return:
    """
    return vector.dot(mat[:-1, :-1])


class _HasAccessorProtocol(Protocol):
    def __getitem__(self, *args, **kwargs) -> T: ...

    def __setitem__(self, key: int, value: T) -> None: ...


class Vec(np.ndarray):
    def __new__(cls, *values, dtype):
        buffer = np.asarray(*values, dtype=dtype)
        obj = super().__new__(cls, shape=(len(*values),), dtype=dtype,
                              buffer=buffer, offset=0, strides=None, order=None)
        return obj

    def norm(self):
        return np.sum(self * self)

    def normalize(self):
        norm = self.norm()
        if norm > 0:
            f = 1. / np.sqrt(norm)
            self[:] *= f

    @classmethod
    def normalizev2(cls, vector: Vec):
        return vector / np.linalg.norm(vector)

    @classmethod
    def mix(cls, a: Vec, b: Vec, mix_value: T) -> Vec:
        return a * (1. - mix_value) + b * mix_value

    def __repr__(self):
        return f"Vec{self.shape[0]}{self.dtype}{repr(tuple(self))}"

    def __mul__(self, other):
        return np.dot(self, other)

    def __abs__(self):
        return np.sqrt(self * self)

    def __pow__(self, x):
        return (self * self) if x == 2 else pow(abs(self), x)

    def __eq__(self, other):
        if not isinstance(other, np.ndarray):
            return False
        return abs(self - other) < 1.e-15

    def __ne__(self, other):
        return not self == other


class _AccessorXYMixin(_HasAccessorProtocol):
    @property
    def x(self) -> T:
        return self[0]

    @x.setter
    def x(self, val: T):
        self[0] = val

    @property
    def y(self) -> T:
        return self[1]

    @y.setter
    def y(self, val: T) -> None:
        self[1] = val


class _AccessorZMixin(_HasAccessorProtocol):
    @property
    def z(self) -> T:
        return self[2]

    @z.setter
    def z(self, val: T) -> None:
        self[2] = val


class Vec3(Vec, _AccessorXYMixin, _AccessorZMixin):
    def __new__(cls, dtype, x: T = 0, y: T = 0, z: T = 0):
        obj = super().__new__(cls, (x, y, z), dtype=dtype)
        return obj

    def mult_vect_matrix(self, mat: np.ndarray):
        if mat.shape != (4, 4):
            raise ValueError(f"This is not a 4x4 Matrix")
        a = self.x * mat[0, 0] + self.y * mat[1, 0] + self.z * mat[2, 0] + mat[3, 0]
        b = self.x * mat[0, 1] + self.y * mat[1, 1] + self.z * mat[2, 1] + mat[3, 1]
        c = self.x * mat[0, 2] + self.y * mat[1, 2] + self.z * mat[2, 2] + mat[3, 2]
        w = self.x * mat[0, 3] + self.y * mat[1, 3] + self.z * mat[2, 3] + mat[3, 3]

        print(a, b, c, w)

        self[0] = a / w
        self[1] = b / w
        self[2] = c / w

    def mult_dir_matrix(self, mat: np.ndarray):
        if mat.shape != (4, 4):
            raise ValueError(f"This is not a 4x4 Matrix")

        a = self[0] * mat[0, 0] + self[1] * mat[1, 0] + self[2] * mat[2, 0]
        b = self[0] * mat[0, 1] + self[1] * mat[1, 1] + self[2] * mat[2, 1]
        c = self[0] * mat[0, 2] + self[1] * mat[1, 2] + self[2] * mat[2, 2]

        self[0] = a
        self[1] = b
        self[2] = c


class Vec2(Vec, _AccessorXYMixin):
    def __new__(cls, dtype, x: T = 0, y: T = 0):
        x = x or 0
        y = y or 0
        obj = super().__new__(cls, (x, y), dtype=dtype)
        return obj


class Vec2f(Vec2):
    def __new__(cls, x, y):
        obj = super().__new__(cls, np.float32, x, y)
        return obj


class Vec2i(Vec2):
    def __new__(cls, x, y):
        obj = super().__new__(cls, np.int, x, y)
        return obj


class Vec3f(Vec3):
    def __new__(cls, x, y, z):
        obj = super().__new__(cls, np.float32, x, y, z)
        return obj


class Vec3i(Vec3):
    def __new__(cls, x, y, z):
        obj = super().__new__(cls, np.int8, x, y, z)
        return obj


if __name__ == "__main__":
    v = Vec3f(1, 2, 3)
    print(v)
    # print(v.z)
    # v.z = 4
    # print(v)
    camera_to_world = np.array([(0.945519, 0, -0.325569, 0),
                                (-0.179534, 0.834209, -0.521403, 0),
                                (0.271593, 0.551447, 0.78876, 0),
                                (4.208271, 8.374532, 17.932925, 1)], dtype=np.dtype(np.float32))
    v.mult_vect_matrix(camera_to_world)
    print(v)
    # vv = Vec3f.normalizev2(v)
    # v.normalize()
    # print(v == vv)
