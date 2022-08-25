from albumentations import BasicTransform
from typing import List, Dict, Any, Tuple, Union, Callable
import numpy as np
from utils.geometry import Boxes_xyxy_n, Box_xywh_n
# #####################################


class Cutout(BasicTransform):
    """ 
        Implement Cutout Transform
        https://arxiv.org/abs/1708.04552

        This is slightly different from the one already implemented in albumentations
        - a box cannot be cut out more than max_ioa (holes cuting out too much are discarded)
        - sizes are now proportional to image size
        - fill value is now randomizable
    """
    def __init__(self, 
        num_holes:      Union[int, Tuple[int, int]] = 1,
        h_size:         Union[float, Tuple[float, float]] = 0.1,
        w_size:         Union[float, Tuple[float, float]] = 0.1,
        max_ioa:        float = 0.3,
        fill_value:     Union[int, Tuple[int, int, int], Tuple[int, int]] = 0,
        always_apply:   bool = False, 
        p:              float = 0.5):

        BasicTransform.__init__(self, always_apply, p)

        self.num_holes  = num_holes
        self.h_size     = h_size
        self.w_size     = w_size
        self.max_ioa    = max_ioa
        self.fill_value = fill_value
    # #####################################

    @property
    def targets(self) -> Dict[str, Callable]:
        """ define mapping between data and processing functions """
        return {'image': self.apply}
    # #####################################

    @property
    def targets_as_params(self) -> List[str]:
        return ('bboxes', )
    # #####################################

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("num_holes", "h_size", "w_size", "max_ioa", "fill_value")
    # #####################################

    @staticmethod
    def ioas(boxes: Boxes_xyxy_n, hole: Box_xywh_n):
        # compute intersection areas
        Ix0 = np.maximum(boxes.X0, hole.x0)
        Ix1 = np.minimum(boxes.X1, hole.x1)
        Iw  = np.maximum(Ix1 - Ix0, 0)
        Iy0 = np.maximum(boxes.Y0, hole.y0)
        Iy1 = np.minimum(boxes.Y1, hole.y1)
        Ih  = np.maximum(Iy1 - Iy0, 0)
        Ia  = Iw * Ih
        # compute box areas
        Ba  = boxes.A
        # compute ratio        
        return Ia / Ba
    # #####################################

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        bboxes      = Boxes_xyxy_n(params['bboxes'])[:, :4]

        # generate hole coordinates
        holes   = []
        for _ in range(self.num_holes):
            # generate the size
            hole_w = self.w_size if isinstance(self.w_size, float) \
                else np.random.uniform(*self.w_size)
            hole_h = self.h_size if isinstance(self.h_size, float) \
                else np.random.uniform(*self.h_size)
            # generate a random position inside the image
            hole_cx = np.random.uniform(hole_w / 2, 1 - hole_w / 2)
            hole_cy = np.random.uniform(hole_h / 2, 1 - hole_h / 2)
            # agg
            hole = Box_xywh_n(hole_cx, hole_cy, hole_w, hole_h)
            # compute per-box IoA 
            ioas = self.ioas(bboxes, hole)
            # use the new hole only if not removing too much of at least one bbox
            if np.all(ioas <= self.max_ioa):
                holes.append(hole)

        return { 'holes': holes }
    # #####################################

    def apply(self, img: np.ndarray, holes, **params) -> np.ndarray:
        h, w = img.shape[:2]
        for hole in holes:
            # select a color
            color = self.get_color()
            # de-normalize hole
            hole = hole.to_xywh(w, h)
            # apply the hole
            img[hole.y0:hole.y1+1, hole.x0:hole.x1+1] = color

        return img
    # #####################################

    def get_color(self, channels: int=3) -> Tuple[int]:
        if isinstance(self.fill_value, int):
            return [self.fill_value] * channels

        elif len(self.fill_value) == 3:
            return self.fill_value

        elif len(self.fill_value) == 2:
            return np.random.uniform(*self.fill_value, channels).astype(int).tolist()

        else:
            raise Exception('Invalid fill_value: %r' % self.fill_value)
    # #####################################
