import numpy as np

# #####################################


class Box_xyxy_n(np.ndarray):

    def __new__(cls, x0, y0, x1, y1):
        return np.array([x0, y0, x1, y1], float).view(cls)

    # ############################

    def __init__(self, x0, y0, x1, y1):
        pass  # only for doc/completion

    # ############################

    def __repr__(self):
        return "<{} x0:{!r} y0:{!r} x1:{!r} y1:{!r}>".format(self.__class__.__name__, self.x0, self.y0, self.x1,
                                                             self.y1)

    # ############################

    @property
    def x0(self):
        return self[0]

    @x0.setter
    def x0(self, value):
        self[0] = value

    # ############################

    @property
    def y0(self):
        return self[1]

    @y0.setter
    def y0(self, value):
        self[1] = value

    # ############################

    @property
    def x1(self):
        return self[2]

    @x1.setter
    def x1(self, value):
        self[2] = value

    # ############################

    @property
    def y1(self):
        return self[3]

    @y1.setter
    def y1(self, value):
        self[3] = value

    # ############################

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2

    @property
    def ar(self):
        return self.h / self.w

    # ############################

    def to_xywh_n(self):
        from .box_xywh_n import Box_xywh_n
        return Box_xywh_n(self.cx, self.cy, self.w, self.h)

    # ############################

    def to_xyxy(self, w, h):
        from .box_xyxy import Box_xyxy
        return (self * (w, h, w, h)).view(Box_xyxy)

    # ############################
