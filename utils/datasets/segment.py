from dataclasses import dataclass

import numpy as np

# #####################################
from ..geometry import Box_xywh_n

# #####################################


@dataclass
class Segment:
    label: int  # class label
    box: Box_xywh_n  # albumentations compatible bbox
    # #####################################

    @classmethod
    def from_line(cls, line: str):
        """ load from a 'space' separated string line:
                cls: int
                x0_n: float
                y0_n: int
                x1_n: float
                y1_n: float
        """
        parts = line.split()
        assert len(parts) == 5, '5 space-separated values expected'

        label = int(parts[0])
        assert label >= 0, 'negative label found'

        box = Box_xywh_n(*[float(part) for part in parts[1:]])
        box = box.to_xyxy_n().clip(0.0, 1.0).to_xywh_n()
        # assert box.x0 >= 0.0, 'non-normalized or out of bounds coordinates: x0 < 0.0'
        # assert box.y0 >= 0.0, 'non-normalized or out of bounds coordinates: y0 < 0.0'
        # assert box.x1 <= 1.0, 'non-normalized or out of bounds coordinates: x1 > 1.0'
        # assert box.y1 <= 1.0, 'non-normalized or out of bounds coordinates: y1 > 1.0'

        return cls(label, box)

    # #####################################
