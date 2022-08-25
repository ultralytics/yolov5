import numpy as np

# #####################################


class Box_xywh_n(np.ndarray):

    def __new__(cls, x, y, w, h):
        return np.array([x, y, w, h], float).view(cls)

    # ############################

    def __init__(self, x, y, w, h):
        pass  # only for doc/completion

    # ############################

    def __repr__(self):
        return f"<{self.__class__.__name__} cx:{self.cx!r} cy:{self.cy!r} w:{self.w!r} h:{self.h!r}>"

    # ############################

    @property
    def cx(self):
        return self[0]

    @cx.setter
    def cx(self, value):
        self[0] = value

    # ############################

    @property
    def cy(self):
        return self[1]

    @cy.setter
    def cy(self, value):
        self[1] = value

    # ############################

    @property
    def w(self):
        return self[2]

    @w.setter
    def w(self, value):
        self[2] = value

    # ############################

    @property
    def h(self):
        return self[3]

    @h.setter
    def h(self, value):
        self[3] = value

    # ############################

    @property
    def ar(self):
        return self.h / self.w

    # ############################

    @property
    def x0(self):
        return self.cx - self.w / 2

    @property
    def y0(self):
        return self.cy - self.h / 2

    @property
    def x1(self):
        return self.cx + self.w / 2

    @property
    def y1(self):
        return self.cy + self.h / 2

    # ############################

    def to_xyxy_n(self):
        from .box_xyxy_n import Box_xyxy_n
        return Box_xyxy_n(self.x0, self.y0, self.x1, self.y1)

    # ############################

    def to_xywh(self, w, h):
        from .box_xywh import Box_xywh
        return (self * (w, h, w, h)).view(Box_xywh)

    # ############################
