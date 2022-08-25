import numpy as np
# #####################################

class Box_xywh(np.ndarray):
    def __new__(cls, x, y, w, h):
        return np.array([x, y, w, h], int).view(cls)
    # ############################

    def __init__(self, x, y, w, h):
        pass # only for doc/completion
    # ############################

    def __repr__(self):
        return "<%s cx:%r cy:%r w:%r h:%r>" % (
            self.__class__.__name__,
            self.cx, self.cy, self.w, self.h
        )
    # ############################

    @property
    def cx(self):        return self[0]
    @cx.setter
    def cx(self, value): self[0] = int(value)
    # ############################

    @property
    def cy(self):        return self[1]
    @cy.setter
    def cy(self, value): self[1] = int(value)
    # ############################

    @property
    def w(self):        return self[2]
    @w.setter
    def w(self, value): self[2] = int(value)
    # ############################

    @property
    def h(self):        return self[3]
    @h.setter
    def h(self, value): self[3] = int(value)
    # ############################

    @property
    def ar(self):       return self.h / self.w
    # ############################

    @property
    def x0(self):       return int(self.cx - self.w / 2)
    @property
    def y0(self):       return int(self.cy - self.h / 2)
    @property
    def x1(self):       return int(self.cx + self.w / 2)
    @property
    def y1(self):       return int(self.cy + self.h / 2)
    # ############################


    def to_xyxy(self):
        from .box_xyxy import Box_xyxy
        return Box_xyxy(self.x0, self.y0, self.x1, self.y1)
    # ############################

    def to_xywh_n(self, w, h):
        from .box_xywh_n import Box_xywh_n
        return (self / (w, h, w, h)).view(Box_xywh_n)
    # ############################