import onnxruntime
import argparse

import cv2
import sys
import numpy as np
import math
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import torch
import torchvision

"""
    Run the inference using onnx runtime, useful debug tool
"""

def get_input_shape(binding_dims):
    if len(binding_dims) == 4:
        return tuple(binding_dims[2:])
    elif len(binding_dims) == 3:
        return tuple(binding_dims[1:])
    else:
        raise ValueError('bad dims of binding %s' % (str(binding_dims)))


class Processor():
    def __init__(self, model, category_num=80, letter_box=False):
        # load onnx engine
        self.ort_session = onnxruntime.InferenceSession(model)

        # get output name
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = []
        for i in range(len(self.ort_session.get_outputs())):
            output_name = self.ort_session.get_outputs()[i].name
            print("output name {}: ".format(i), output_name)
            output_shape = self.ort_session.get_outputs()[i].shape
            print("output shape {}: ".format(i), output_shape)
            self.output_names.append(output_name)

        self.input_shape = get_input_shape(self.ort_session.get_inputs()[0].shape)
        print('---self.input_shape: ', self.input_shape)


        self.letter_box = letter_box
        # post processing config
        filters = (category_num + 5) * 3
        self.output_shapes = [
            (1, 3, 80, 80, 85),
            (1, 3, 40, 40, 85),
            (1, 3, 20, 20, 85)
        ]
        self.strides = np.array([8., 16., 32.])
        anchors = np.array([
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]],
        ])
        self.nl = len(anchors)
        self.nc = category_num  # classes
        self.no = self.nc + 5  # outputs per anchor
        self.na = len(anchors[0])
        a = anchors.copy().astype(np.float32)
        a = a.reshape(self.nl, -1, 2)
        self.anchors = a.copy()
        self.anchor_grid = a.copy().reshape(self.nl, 1, -1, 1, 1, 2)

    def detect(self, img, letter_box=None):
        """Detect objects in the input image."""
        letter_box = self.letter_box if letter_box is None else letter_box
        resized = self.pre_process(img, self.input_shape, letter_box)

        outputs = self.inference(resized)
        # reshape from flat to (1, 3, x, y, 85)
        reshaped = []
        for output, shape in zip(outputs, self.output_shapes):
            reshaped.append(output.reshape(shape))
        return reshaped

    def pre_process(self, img, input_shape, letter_box=False):
        """Preprocess an image before TRT YOLO inferencing.
        """
        if letter_box:
            img_h, img_w, _ = img.shape
            new_h, new_w = input_shape[0], input_shape[1]
            offset_h, offset_w = 0, 0
            if (new_w / img_w) <= (new_h / img_h):
                new_h = int(img_h * new_w / img_w)
                offset_h = (input_shape[0] - new_h) // 2
            else:
                new_w = int(img_w * new_h / img_h)
                offset_w = (input_shape[1] - new_w) // 2
            resized = cv2.resize(img, (new_w, new_h))
            img = np.full((input_shape[0], input_shape[1], 3), 114, dtype=np.uint8)
            img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
        else:
            img = cv2.resize(img, (input_shape[1], input_shape[0]))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        return img

    def inference(self, img):
        img = img[np.newaxis, :, :, :]

        # forward model
        res = self.ort_session.run(self.output_names, {self.input_name: img})

        # Return only the host outputs.
        return [item for item in res]

    def post_process(self, outputs, img_shape, conf_thres=0.5, iou_thres=0.6):
        """
        Transforms raw output into boxes, confs, classes
        Applies NMS thresholding on bounding boxes and confs
        Parameters:
            output: raw output tensor
        Returns:
            boxes: x1,y1,x2,y2 tensor (dets, 4)
            confs: class * obj prob tensor (dets, 1)
            classes: class type tensor (dets, 1)
        """
        scaled = []
        grids = []
        for out in outputs:
            print('out.shape: ', out.shape)
            print('out: ', out)
            out = self.sigmoid_v(out)

            print('sigmoid_v out.shape: ', out.shape)
            print('sigmoid_v out: ', out)
            _, _, width, height, _ = out.shape
            grid = self.make_grid(width, height)
            grids.append(grid)
            scaled.append(out)
        z = []
        for out, grid, stride, anchor in zip(scaled, grids, self.strides, self.anchor_grid):
            _, _, width, height, _ = out.shape
            out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
            out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor

            out = out.reshape((1, 3 * width * height, 85))
            z.append(out)

        pred = np.concatenate(z, 1)

        # Use Pytorch to do the post-process
        det_t = self.non_max_suppression(torch.from_numpy(pred), conf_thres=conf_thres,
                                         iou_thres=iou_thres, multi_label=True)[0]

        self.scale_coords(self.input_shape, det_t[:, :4], img_shape)
        return det_t

    def make_grid(self, nx, ny):
        """
        Create scaling tensor based on box location
        Source: https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
        Arguments
            nx: x-axis num boxes
            ny: y-axis num boxes
        Returns
            grid: tensor of shape (1, 1, nx, ny, 80)
        """
        nx_vec = np.arange(nx)
        ny_vec = np.arange(ny)
        yv, xv = np.meshgrid(ny_vec, nx_vec)
        grid = np.stack((yv, xv), axis=2)
        grid = grid.reshape(1, 1, ny, nx, 2)
        return grid

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_v(self, array):
        return np.reciprocal(np.exp(-array) + 1.0)

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def clip_coords(self, boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False,
                            labels=()):
        """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

        return output

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def cli():
    desc = 'Run TensorRT yolov5 visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default='./weights/yolov5s-ReLU-max-512.onnx', help='onnx runtime file path', required=False)
    parser.add_argument('-i', '--image', default='./data/images/bus.jpg', help='image file path', required=False)
    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = cli()

    # setup processor and visualizer
    processor = Processor(model=args.model, letter_box=True)

    img = cv2.imread(args.image)

    # inference
    output = processor.detect(img)

    # final results
    pred = processor.post_process(output, img.shape, conf_thres=0.5)

    print('Detection result: ')
    for item in pred.tolist():
        print(item)


if __name__ == '__main__':
    main()
