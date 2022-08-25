import numpy as np
# #####################################

class Point(np.ndarray):
    def __new__(cls, x, y):
        return np.array([x, y], int).view(cls)
    # ############################

    def __init__(self, x, y):
        pass # only for doc/completion
    # ############################

    def __repr__(self):
        return "<%s x:%r y:%r>" % (self.__class__.__name__, self.x, self.y)
    # ############################

    @property
    def x(self):        return self[0]
    @x.setter
    def x(self, value): self[0] = int(value)
    # ############################

    @property
    def y(self):        return self[1]
    @y.setter
    def y(self, value): self[1] = int(value)
    # ############################

    def normalize(self, w, h):
        from .point_n import Point_n
        return (self / (w, h)).view(Point_n)
    # ############################
    