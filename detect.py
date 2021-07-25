import argparse
import time
import colorsys
import yaml
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.yolo_mask import Model
import numpy as np
from models.experimental import attempt_load, single_model_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect():
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = False  # half precision only supported on CUDA
    model = Model(opt.cfg, ch=3, nc=1, anchors=opt.hyp.get('anchors'), training=False).to(device)
    model = single_model_load(model, weights, device)

    # Load model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        s = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        output, features_map = model(img, augment=opt.augment)

        # Apply NMS
        dets = non_max_suppression(output[0], opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)[0]
        im0 = im0s

        # Process detections

        if len(dets):
            # Rescale boxes from img_size to im0 size
            mask_rois = dets[:, :4].clone().detach().cpu().numpy()
            mask_rois[:, ::2] /= img.shape[2:][1]
            mask_rois[:, 1::2] /= img.shape[2:][0]
            class_ids = dets[:, -1].clone().cpu().detach().numpy()
            all_scores = dets[:, -2].clone().cpu().detach().numpy().reshape(-1, 1)

            dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], im0.shape).round()

            rois = dets[:, :4].clone().cpu().detach().numpy()
            mask_rois[:, [0, 1, 2, 3]] = mask_rois[:, [1, 0, 3, 2]] #xyxy -> yxyx

            class_ids = np.array(class_ids, dtype='float32')
            class_ids = np.reshape(class_ids, (-1, 1))

            all_scores = np.array(all_scores, dtype='float32')
            all_scores = np.reshape(all_scores, (-1, 1))

            rois = torch.tensor(rois, device='cuda').float()
            mask_rois = torch.tensor(mask_rois, device='cuda').float()

            all_scores = torch.tensor(all_scores, device='cuda').float()
            class_ids = torch.tensor(class_ids, device='cuda').float()

            detections = torch.cat([rois, all_scores, class_ids], dim=1)
            detections = detections.data.cpu().numpy()

            ft_map = [features_map[0][0].type(torch.float32)]
            mrcnn_mask = model.mask_model(ft_map, mask_rois, list(img.shape[2:]))
            mrcnn_mask = mrcnn_mask.data.cpu().numpy()
            mrcnn_mask = np.moveaxis(mrcnn_mask, 1, -1)

            final_rois, final_class_ids, final_masks, final_scores = unmold_detections(detections, mrcnn_mask)
            result = {
                "rois": final_rois,
                "class_ids": final_class_ids,
                "masks": final_masks,
                "scores": final_scores
            }
            display_instances_cv2(im0, result['rois'], result['masks'], result['class_ids'], result['scores'],
                                        class_name=['obj'])
        print('time is ', time.time()-s)
def unmold_mask(mask, bbox):
    """
    :param mask: mask image (28x28x1)
    :param bbox: xyxy related to original image's shape
    :return:
    """
    threshold = 0.4
    x1, y1, x2, y2 = bbox

    mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation= cv2.INTER_LINEAR).astype(np.float32)
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # cv2.imshow('mask', mask* 255)
    # cv2.waitKey()
    # Put the mask in the right location.
    #full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    #full_mask[y1: y2, x1: x2] = mask
    return mask

def unmold_detections(detections, mrcnn_mask):
    N = detections.shape[0]
    boxes = detections[:N, :4]
    class_ids = detections[:N, 5].astype(np.int32) + 1
    scores = detections[:N, 4].astype(np.float32)
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]

    boxes = np.round(boxes).astype(np.int32)

    full_masks = []

    for i in range(N):
        full_mask = unmold_mask(masks[i], boxes[i])
        full_masks.append(full_mask)

    return boxes, class_ids, full_masks, scores

def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances_cv2(image, boxes, masks, class_ids, scores, show_scores= False, show_mask= True,
                          show_bbox= True, colors= None, class_name= None, draw_boundary= False):

    # Number of instances
    N = boxes.shape[0]

    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == len(masks) == class_ids.shape[0]

    # Generate random colors
    colors = colors or random_colors(N)
    masked_image = image.astype(np.uint8).copy()
    overlay = masked_image.copy()

    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            continue
        x1, y1, x2, y2 = boxes[i]

        if show_bbox:
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), color= [255, 0, 255], thickness= 2)

        # Label
        if class_name is not None:
            str_class = class_name[class_ids[i] - 1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(masked_image, str_class, (x1, y1), font, 1, [0, 0, 255], 2, cv2.LINE_AA)

        # Scores:
        if show_scores:
            str_score = '{0:.2f}'.format(scores[i])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(masked_image, str_score, (x2, y1), font, 1, [255, 0, 0], 2, cv2.LINE_AA)

        # Mask
        if show_mask:
            mask = np.array(masks[i])
            a1, a2 = np.where(mask == 1)
            b = np.asarray(list(zip(a2, a1)), dtype= np.float32)
            b = np.reshape(b, (-1, 2))

            add_array = [x1, y1]
            new_point_mask = b + add_array
            verts = np.array(new_point_mask).reshape((1, -1, 2)).astype(np.int32)

            if not draw_boundary:
                cv2.drawContours(overlay, verts, -1, 255 * np.array(color), 1)
            else:
                convex = cv2.convexHull(verts, False)
                convex = np.reshape(convex, (1, -1, 2))
                cv2.drawContours(masked_image, convex, -1, 255 * np.array(color), 2)

    if not draw_boundary:
        cv2.addWeighted(overlay, 0.6, masked_image, 0.4, 0, masked_image)

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('result', 1200, 860)
    cv2.imshow('result', masked_image)
    end = time.time()
    cv2.waitKey(0)

    return end


if __name__ == '__main__':
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/yolo_mask/model_3000.pth', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r'F:\data\Unknown Segmentation Tool10\DetectImages', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.cfg = 'models/yolov5x.yaml'
    opt.hyp = 'runs/train/exp9/hyp.yaml'
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
        opt.hyp = hyp
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
