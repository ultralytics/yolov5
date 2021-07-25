import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import roi_align.crop_and_resize_cpu as crop_and_resize_cpu
if torch.cuda.is_available():
    import roi_align.crop_and_resize_gpu as crop_and_resize_gpu



class CropAndResizeFunction(Function):

    @staticmethod
    def forward(ctx, image, boxes, box_ind, crop_height, crop_width, extrapolation_value=0):
        ctx.crop_height = crop_height
        ctx.crop_width = crop_width
        ctx.extrapolation_value = extrapolation_value
        crops = torch.zeros_like(image)

        if image.is_cuda:
            crop_and_resize_gpu.forward(
                image, boxes, box_ind,
                ctx.extrapolation_value, ctx.crop_height, ctx.crop_width, crops)
        else:
            crop_and_resize_cpu.forward(
                image, boxes, box_ind,
                ctx.extrapolation_value, ctx.crop_height, ctx.crop_width, crops)

        # save for backward
        ctx.im_size = image.size()
        ctx.save_for_backward(boxes, box_ind)

        return crops

    @staticmethod
    def backward(ctx, grad_outputs):
        boxes, box_ind = ctx.saved_tensors

        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros_like(grad_outputs).resize_(*ctx.im_size)

        if grad_outputs.is_cuda:
            crop_and_resize_gpu.backward(
                grad_outputs, boxes, box_ind, grad_image
            )
        else:
            crop_and_resize_cpu.backward(
                grad_outputs, boxes, box_ind, grad_image
            )

        return grad_image, None, None, None, None, None


class CropAndResize(nn.Module):
    """
    Crop and resize ported from tensorflow
    See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    """

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(CropAndResize, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        return CropAndResizeFunction.apply(image, boxes, box_ind, self.crop_height, self.crop_width, self.extrapolation_value)
