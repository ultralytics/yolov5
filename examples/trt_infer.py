import sys

import cv2

sys.path.append('../')
import random
import time
from collections import OrderedDict, namedtuple

import numpy as np
import tensorrt as trt
import torch
from PIL import Image

from utils.augmentations import letterbox

names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(names)}

w = './yolov5s_nms_fp16.engine'
image_path = './image1.jpg'
device = torch.device('cuda:0')

# Infer TensorRT Engine
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, namespace="")
with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
    model = runtime.deserialize_cuda_engine(f.read())
bindings = OrderedDict()
fp16 = False  # default updated below
for index in range(model.num_bindings):
    name = model.get_binding_name(index)
    dtype = trt.nptype(model.get_binding_dtype(index))
    shape = tuple(model.get_binding_shape(index))
    data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
    bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    if model.binding_is_input(index) and dtype == np.float16:
        fp16 = True
binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
context = model.create_execution_context()

image = cv2.imread(image_path)
image, ratio, dwdh = letterbox(image, auto=False)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_copy = image.copy()

image = image.transpose((2, 0, 1))
image = np.expand_dims(image, 0)
image = np.ascontiguousarray(image)
im = torch.from_numpy(image).to(device)
im = im.float()
im /= 255

# warmup for 10 times
for _ in range(10):
    tmp = torch.randn(1, 3, 640, 640).to(device)
    binding_addrs['images'] = int(tmp.data_ptr())
    context.execute_v2(list(binding_addrs.values()))

start = time.perf_counter()
binding_addrs['images'] = int(im.data_ptr())
context.execute_v2(list(binding_addrs.values()))
print(f'Cost {time.perf_counter()-start} s')

nums = bindings['num_dets'].data
boxes = bindings['det_boxes'].data
scores = bindings['det_scores'].data
classes = bindings['det_classes'].data

print(nums)
print(boxes)
print(scores)
print(classes)

num = int(nums[0][0])
box_img = boxes[0, :num].round().int()
score_img = scores[0, :num]
clss_img = classes[0, :num]
for i, (box, score, clss) in enumerate(zip(box_img, score_img, clss_img)):
    name = names[clss]
    color = colors[name]
    cv2.rectangle(image_copy, box[:2].tolist(), box[2:].tolist(), color, 2)
    cv2.putText(image_copy,
                name, (int(box[0]), int(box[1]) - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75, [225, 255, 255],
                thickness=2)

Image.fromarray(image_copy).show()
