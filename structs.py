from collections import namedtuple
from enum import IntEnum, auto
import numpy as np


class Coordinate(namedtuple('Coordinate', ('y', 'x'))):
    __slots__ = ()

    @staticmethod
    def _get_one(value):
        if value > 0:
            return 1
        elif value < 0:
            return -1
        else:
            return 0

    def __add__(self, other):
        return Coordinate(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other):
        return Coordinate(x=self.x - other.x, y=self.y - other.y)

    def get_ones(self):
        return Coordinate(x=self._get_one(self.x), y=self._get_one(self.y))


class Direction(IntEnum):
    NW = auto()
    W = auto()
    SW = auto()
    N = auto()
    C = auto()
    S = auto()
    NE = auto()
    E = auto()
    SE = auto()

    def get_coordinate(self) -> Coordinate:
        aux = self.value - 1
        return Coordinate(y=(aux % 3 - 1), x=(aux // 3 - 1))

    def rotate(self, steps: int = 1):
        to_right = (4, 1, 2, 7, 5, 3, 8, 9, 6)
        to_left = (2, 3, 6, 1, 5, 9, 4, 7, 8)

        new_direction = self

        if steps == 0:
            return self

        if steps < 0:
            while steps < 0:
                new_direction = Direction(to_left[new_direction - 1])
                steps += 1
        elif steps > 0:
            while steps > 0:
                new_direction = Direction(to_right[new_direction - 1])
                steps -= 1

        return new_direction

    @classmethod
    def get_random_direction(cls):
        while (direction := cls(np.random.randint(1, 10))) == cls.C:
            pass

        return direction

    @classmethod
    def from_coordinate(cls, coordinate: Coordinate):
        return cls((coordinate.x + 1) * 3 + (coordinate.y + 2))
