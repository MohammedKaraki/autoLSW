import numpy as np
from fractions import Fraction


class Coords:
    def __init__(self, x, y, z):
        self.coords_arr = np.array([x, y, z], dtype=Fraction)

    @classmethod
    def from_array(cls, arr):
        x, y, z = arr
        return cls(x, y, z)

    def all_ints(self):
        return np.all(np.floor(self.coords_arr) == self.coords_arr)

    def __sub__(self, rhs):
        return Coords.from_array(self.coords_arr - rhs.coords_arr)
    def __add__(self, rhs):
        return Coords.from_array(self.coords_arr + rhs.coords_arr)
    def __eq__(self, rhs):
        return np.all(self.coords_arr == rhs.coords_arr)

    def __hash__(self):
        return 0

    def __str__(self):
        return "(" + ", ".join(str(a) for a in self.coords_arr) + ")"

