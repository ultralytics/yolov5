import numpy as np

# #####################################


class Box_xyxy(np.ndarray):

    def __new__(cls, x0, y0, x1, y1):
        return np.array([x0, y0, x1, y1], int).view(cls)

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
        self[0] = int(value)
    # ############################

    @property
    def y0(self):
        return self[1]

    @y0.setter
    def y0(self, value):
        self[1] = int(value)
    # ############################

    @property
    def x1(self):
        return self[2]

    @x1.setter
    def x1(self, value):
        self[2] = int(value)
    # ############################

    @property
    def y1(self):
        return self[3]

    @y1.setter
    def y1(self, value):
        self[3] = int(value)
    # ############################

    @property
    def w(self):
        return int(self.x1 - self.x0 + 1)

    @property
    def h(self):
        return int(self.y1 - self.y0 + 1)

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

    def to_xywh(self):
        from .box_xywh import Box_xywh
        return Box_xywh(self.cx, self.cy, self.w, self.h)

    # ############################

    def to_xyxy_n(self, w, h):
        from .box_xyxy_n import Box_xyxy_n
        return (self / (w, h, w, h)).view(Box_xyxy_n)

    # ############################
