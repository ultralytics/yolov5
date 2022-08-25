import numpy as np

# #####################################


class Boxes_xywh(np.ndarray):

    def __new__(cls, boxes_xywh):
        if len(boxes_xywh):
            if isinstance(boxes_xywh, np.ndarray):
                return np.vstack(boxes_xywh).view(cls)
            else:
                return np.array(boxes_xywh).view(cls)
        else:
            return np.zeros((0, 4)).view(cls)

    # ############################

    def __init__(self, boxes_xywh):
        pass  # only for doc/completion

    # ############################

    def __repr__(self):
        return "<%s #%d>" % (self.__class__.__name__, len(self))

    # ############################

    @property
    def CX(self):
        return self[:, 0]

    @property
    def CY(self):
        return self[:, 1]

    @property
    def W(self):
        return self[:, 2]

    @property
    def H(self):
        return self[:, 3]
    # ############################

    @CX.setter
    def CX(self, value):
        self[:, 0] = value

    @CY.setter
    def CY(self, value):
        self[:, 1] = value

    @W.setter
    def W(self, value):
        self[:, 2] = value

    @H.setter
    def H(self, value):
        self[:, 3] = value
    # ############################

    @property
    def A(self):
        return self.H * self.W

    @property
    def AR(self):
        return self.H / self.W

    @property
    def X0(self):
        return self.CX - (self.W / 2)

    @property
    def Y0(self):
        return self.CY - (self.H / 2)

    @property
    def X1(self):
        return self.CX + (self.W / 2)

    @property
    def Y1(self):
        return self.CY + (self.H / 2)

    # ############################

    def to_xyxy(self):
        from .boxes_xyxy import Boxes_xyxy
        return np.vstack((self.X0, self.Y0, self.X1, self.Y1)).T.view(Boxes_xyxy)

    # ############################

    def to_xywh_n(self, w, h):
        from .boxes_xywh_n import Boxes_xywh_n
        return (self / (w, h, w, h)).astype(float).view(Boxes_xywh_n)

    # ############################
