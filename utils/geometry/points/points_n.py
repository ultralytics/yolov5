import numpy as np

# #####################################


class Points_n(np.ndarray):

    def __new__(cls, points_n):
        return np.array(points_n, float).view(cls)

    # ############################

    def __init__(self, points_n):
        pass  # only for doc/completion

    # ############################

    def __repr__(self):
        return "<%s #%d>" % (self.__class__.__name__, len(self))

    # ############################

    @property
    def X(self):
        return self[0]

    @property
    def Y(self):
        return self[1]
    # ############################

    @X.setter
    def X(self, value):
        self[0] = value

    @Y.setter
    def Y(self, value):
        self[1] = value

    # ############################

    def denormalize(self, w, h):
        from .points import Points
        return (self * (w, h)).view(Points)

    # ############################
