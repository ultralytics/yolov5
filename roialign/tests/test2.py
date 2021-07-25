import numpy as np
import torch
from torch.autograd import Variable

from roi_align.roi_align import RoIAlign


def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


# the data you want
is_cuda = False
image_data = np.tile(np.arange(7, dtype=np.float32), 7).reshape(7, 7)
image_data = image_data[np.newaxis, np.newaxis]
boxes_data = np.asarray([[0, 0, 3, 3]], dtype=np.float32)
box_index_data = np.asarray([0], dtype=np.int32)

image_torch = to_varabile(image_data, requires_grad=True, is_cuda=is_cuda)
boxes = to_varabile(boxes_data, requires_grad=False, is_cuda=is_cuda)
box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=is_cuda)

# set transform_fpcoor to False is the crop_and_resize
roi_align = RoIAlign(3, 3, transform_fpcoor=True)
print(roi_align(image_torch, boxes, box_index))