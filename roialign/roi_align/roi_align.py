import torch
from torch import nn

from .crop_and_resize import CropAndResizeFunction, CropAndResize


class RoIAlign(nn.Module):

    def __init__(self, crop_height, crop_width, extrapolation_value=0, transform_fpcoor=True):
        super(RoIAlign, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value
        self.transform_fpcoor = transform_fpcoor

    def forward(self, featuremap, boxes, box_ind):
        """
        RoIAlign based on crop_and_resize.
        See more details on https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L301
        :param featuremap: NxCxHxW
        :param boxes: Mx4 float box with (x1, y1, x2, y2) **without normalization**
        :param box_ind: M
        :return: MxCxoHxoW
        """
        x1, y1, x2, y2 = torch.split(boxes, 1, dim=1)
        image_height, image_width = featuremap.size()[2:4]

        if self.transform_fpcoor:
            spacing_w = (x2 - x1) / float(self.crop_width)
            spacing_h = (y2 - y1) / float(self.crop_height)

            nx0 = (x1 + spacing_w / 2 - 0.5) / float(image_width - 1)
            ny0 = (y1 + spacing_h / 2 - 0.5) / float(image_height - 1)
            nw = spacing_w * float(self.crop_width - 1) / float(image_width - 1)
            nh = spacing_h * float(self.crop_height - 1) / float(image_height - 1)

            boxes = torch.cat((ny0, nx0, ny0 + nh, nx0 + nw), 1)
        else:
            x1 = x1 / float(image_width - 1)
            x2 = x2 / float(image_width - 1)
            y1 = y1 / float(image_height - 1)
            y2 = y2 / float(image_height - 1)
            boxes = torch.cat((y1, x1, y2, x2), 1)

        boxes = boxes.detach().contiguous()
        box_ind = box_ind.detach()
        return CropAndResizeFunction.apply(featuremap, boxes, box_ind, self.crop_height, self.crop_width, self.extrapolation_value)
