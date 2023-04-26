import cv2
import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt

coco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

class Visualizer():
    def __init__(self):
        self.color_list = self.gen_colors(coco)
    
    def gen_colors(self, classes):
        """
            generate unique hues for each class and convert to bgr
            classes -- list -- class names (80 for coco dataset)
            -> list
        """
        hsvs = []
        for x in range(len(classes)):
            hsvs.append([float(x) / len(classes), 1., 0.7])
        random.seed(1234)
        random.shuffle(hsvs)
        rgbs = []
        for hsv in hsvs:
            h, s, v = hsv
            rgb = colorsys.hsv_to_rgb(h, s, v)
            rgbs.append(rgb)

        bgrs = []
        for rgb in rgbs:
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            bgrs.append(bgr)
        return bgrs


    def draw_results(self, img, boxes, confs, classes):
        overlay = img.copy()
        final = img.copy()
        for box, conf, cls in zip(boxes, confs, classes):
            # draw rectangle
            x1, y1, x2, y2 = box
            color = self.color_list[int(cls)]
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)

        cv2.addWeighted(overlay, 0.5, final, 1 - 0.5, 0, final)

        cv2.imwrite('./box_grid.jpg', final)
        return final
