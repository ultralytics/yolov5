import numpy as np
# #####################################

class Boxes_xyxy_n(np.ndarray):
    def __new__(cls, boxes_xyxy_n):
        if len(boxes_xyxy_n):
            if isinstance(boxes_xyxy_n, np.ndarray):
                return np.vstack(boxes_xyxy_n).view(cls)
            else:
                return np.array(boxes_xyxy_n).view(cls)
        else:
            return np.zeros((0, 4)).view(cls)
    # ############################

    def __init__(self, boxes_xyxy_n):
        pass # only for doc/completion
    # ############################

    def __repr__(self):
        return "<%s #%d>" % (self.__class__.__name__, len(self))
    # ############################

    @property
    def X0(self):   return self[:,0]
    @property
    def Y0(self):   return self[:,1]
    @property
    def X1(self):   return self[:,2]
    @property
    def Y1(self):   return self[:,3]
    # ############################

    @X0.setter
    def X0(self, value):   self[:,0] = value
    @Y0.setter
    def Y0(self, value):   self[:,1] = value
    @X1.setter
    def X1(self, value):   self[:,2] = value
    @Y1.setter
    def Y1(self, value):   self[:,3] = value
    # ############################

    @property
    def A(self):            return self.H * self.W
    @property
    def W(self):            return self.X1 - self.X0
    @property
    def H(self):            return self.Y1 - self.Y0
    @property
    def CX(self):           return (self.X0 + self.X1) / 2
    @property
    def CY(self):           return (self.Y0 + self.Y1) / 2
    @property
    def AR(self):           return self.H / self.W
    # ############################

    def to_xywh_n(self):
        from .boxes_xywh_n import Boxes_xywh_n
        return np.vstack((self.CX, self.CY, self.W, self.H)).T.view(Boxes_xywh_n)
    # ############################

    def to_xyxy(self, w, h):
        from .boxes_xyxy import Boxes_xyxy
        return (self * (w, h, w, h)).astype(int).view(Boxes_xyxy)
    # ############################