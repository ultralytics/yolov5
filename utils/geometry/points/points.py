import numpy as np

# #####################################


class Points(np.ndarray):

    def __new__(cls, points):
        return np.array(points, int).view(cls)

    # ############################

    def __init__(self, points):
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

    def normalize(self, w, h):
        from .points_n import Points_n
        return (self / (w, h)).view(Points_n)

    # ############################
