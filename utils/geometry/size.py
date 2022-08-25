import numpy as np

# #####################################


class Size(np.ndarray):

    def __new__(cls, w, h):
        return np.array([w, h]).view(cls)

    # ############################

    def __init__(self, w, h):
        pass  # only for doc/completion

    # ############################

    def __repr__(self):
        return "<{} w:{!r} h:{!r}>".format(self.__class__.__name__, self.w, self.h)

    # ############################

    @property
    def w(self):
        return self[0]

    @property
    def h(self):
        return self[1]
    # ############################

    @w.setter
    def w(self, value):
        self[0] = value

    @h.setter
    def h(self, value):
        self[1] = value
    # ############################

    @property
    def ar(self):
        return self.h / self.w

    # ############################
