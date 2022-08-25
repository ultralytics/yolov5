import numpy as np

# #####################################


class Point_n(np.ndarray):

    def __new__(cls, x, y):
        return np.array([x, y], float).view(cls)

    # ############################

    def __init__(self, x, y):
        pass  # only for doc/completion

    # ############################

    def __repr__(self):
        return f"<{self.__class__.__name__} x:{self.x!r} y:{self.y!r}>"

    # ############################

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = int(value)

    # ############################

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = int(value)

    # ############################

    def denormalize(self, w, h):
        from .point import Point
        return (self * (w, h)).view(Point)

    # ############################
