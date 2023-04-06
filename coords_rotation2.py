# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import datetime
import glob
import math
import os
import platform
import queue
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from cv2 import dnn_superres
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

path_crops = '/home/gabriel/projeto/yolov5/runs/detect/crops/*'
pathResolution = os.path.abspath('./modelsResolution/ESPCN_x4.pb')


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):

    #Initiate sr object

    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(pathResolution)
    sr.setModel('espcn', 4)

    #Reset center Point Apple
    center_point_apple = None
    cropped_img = None
    cropped_img_heigth = None
    rotation = None
    prev_img = None

    #Initiate result Image
    result = Image.new('RGB', (320, 320), (255, 255, 255, 0))
    height = 0
    whidth = 0

    #Reset coordinates of calice
    tlx_c = 0
    tly_c = 0
    brx_c = 0
    bry_c = 0

    #Reset coordinates of pedunculo
    tlx_p = 0
    tly_p = 0
    brx_p = 0
    bry_p = 0

    #Reset coordinates of apple detection
    tlx_a = 0
    tly_a = 0
    brx_a = 0
    bry_a = 0

    #Reset coordinates of def_rotten detection
    tlx_dr = 0
    tly_dr = 0
    brx_dr = 0
    bry_dr = 0

    #Reset coordinates of def_old detection
    tlx_o = 0
    tly_o = 0
    brx_o = 0
    bry_o = 0

    #Resetcoordinates of def_color detection
    tlx_cl = 0
    tly_cl = 0
    brx_cl = 0
    bry_cl = 0

    #Reset coordinates of def_cut detection
    tlx_ct = 0
    tly_ct = 0
    brx_ct = 0
    bry_ct = 0

    #Reset previous center points x-coordinates
    prev_center_point_c_x = None
    prev_center_point_p_x = None
    prev_center_point_dr_x = None
    prev_center_point_o_x = None
    prev_center_point_cl_x = None
    prev_center_point_ct_x = None

    #Reset previous center points y-coordinates
    prev_center_point_c_y = None
    prev_center_point_p_y = None
    prev_center_point_dr_y = None
    prev_center_point_o_y = None
    prev_center_point_cl_y = None
    prev_center_point_ct_y = None

    #Reset previous angles
    prev_calice_angle = None
    prev_pedunculo_angle = None
    prev_rotten_angle = None
    prev_old_angle = None
    prev_color_angle = None
    prev_cut_angle = None

    #queue to store areas of rotten detections
    q = queue.Queue()

    #queue to store areas of old defect detections
    qo = queue.Queue()

    #queue to store areas of color defect detections
    qcl = queue.Queue()

    #queue to store areas of cut defect detections
    qct = queue.Queue()

    #queue to store distances
    cpoints = queue.Queue()

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir = Path(project)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0

            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions

        for i, det in enumerate(pred):  # per image
            seen += 1

            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            im_base = im0
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                bboxes = det[:, :4]

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    #print("\n",s)
                    #to get the coordinates of the bounding boxes
                    c1 = (int(xyxy[0]), int(xyxy[1]))  # top-left (tl) corner of the bounding box
                    c2 = (int(xyxy[2]), int(xyxy[3]))  # bottom-right (br) corner of the bounding box

                    #------------------------------------------------IF DETECTED APPLE------------------------------------------------------------#

                    #this class must be verified first in order to process the others
                    if cls == 4:

                        #print("\n",s)
                        #print("Apple detected")

                        #store the coordinate points of the bounding box of the detected class "apple" in a queue
                        cpoints.put(c1[0])
                        cpoints.put(c1[1])
                        cpoints.put(c2[0])
                        cpoints.put(c2[1])

                        #gets the coordinate points of the bounding box of the detected class "apple" from the queue and passes to variables
                        tlx_a = cpoints.get()
                        tly_a = cpoints.get()
                        brx_a = cpoints.get()
                        bry_a = cpoints.get()

                        vertices = np.array([[tlx_a, tly_a], [brx_a, tly_a], [brx_a, bry_a], [tlx_a, bry_a]],
                                            dtype=np.int32)

                        n_pedunculo = 0
                        n_calice = 0
                        for *xyxy, conf, cls in reversed(det):
                            c1 = (int(xyxy[0]), int(xyxy[1]))  # top-left (tl) corner of the bounding box
                            c2 = (int(xyxy[2]), int(xyxy[3]))  # bottom-right (br) corner of the bounding box

                            # Create mask of zeros with same size as input image
                            maskA = np.zeros_like(im0[:, :, 0])

                            # Draw polygon on mask using vertices
                            #cv2.fillPoly(maskA, [vertices], (255,255,255))

                            # Mask input image with polygon mask
                            masked_img = cv2.bitwise_and(im0, im0, mask=maskA)

                            x, y, w, h = cv2.boundingRect(vertices)

                            # Crop the masked image to the size of the bounding box
                            cropped_img = im0[y:y + h, x:x + w]

                            # Show the image
                            #cv2.imshow('Base Image', im_base)
                            #cv2.waitKey(0)
                            #------------------------------------------------IF DETECTED CALICE------------------------------------------------------------#
                            if cls == 6 and n_pedunculo == 0 and n_calice == 0:
                                n_calice += 1
                                print('\n', s)
                                print('\033[1mCALICE DETECTED\033[0m')
                                #store the coordinate points of the bounding box of the detected class "calice" in a queue
                                cpoints.put(c1[0])
                                cpoints.put(c1[1])
                                cpoints.put(c2[0])
                                cpoints.put(c2[1])

                                #gets the coordinate points of the bounding box of the detected class "calice" from the queue and passes to variables
                                tlx_c = cpoints.get()
                                tly_c = cpoints.get()
                                brx_c = cpoints.get()
                                bry_c = cpoints.get()

                                #Compute center point of apple plane detection
                                longitude = bry_a - tly_a
                                latitude = brx_a - tlx_a
                                diff_LoLa_c = longitude - latitude

                                #Compute center point of pedunculo detection
                                center_point_c = round((tlx_c + brx_c) / 2), round((tly_c + bry_c) / 2)
                                print(center_point_c, 'center point c')
                                center_point_apple = round((tlx_a + brx_a) / 2), round((tly_a + bry_a) / 2)

                                if diff_LoLa_c > 75:
                                    if center_point_apple[0] > center_point_c[0]:
                                        a = (center_point_apple[0] - center_point_c[0] / 2)
                                        center_point_apple = round(center_point_c[0] - a), round((tly_a + bry_a) / 2)
                                    else:
                                        a = ((center_point_c[0] - center_point_apple[0]) / 2)
                                        center_point_apple = round(center_point_c[0] + a), round((tly_a + bry_a) / 2)

                                #-----------------APPLE Operations--------------------------------#
                                #Compute center point of apple plane detection

                                #print("Center point of apple", center_point_apple)
                                cv2.circle(im0, center_point_apple, 5, (255, 0, 0), 2)

                                #-----------------CALICE Operations--------------------------------#
                                #print("Center point of calice detection: ", center_point_c)
                                cv2.circle(im0, center_point_c, 5, (255, 0, 0), 2)

                                current_center_point_c_x = (tlx_c + brx_c) / 2
                                current_center_point_c_y = (tly_c + bry_c) / 2

                                #Compute distance between center point of apple plane and center point of calice defection
                                distance_c = math.dist((center_point_apple[0], center_point_apple[1]),
                                                       (center_point_c[0], center_point_c[1]))

                                distancia = [
                                    center_point_apple[0] - center_point_c[0],
                                    center_point_apple[1] - center_point_c[1]]
                                distancia_1 = [
                                    center_point_apple[0] + distancia[0], center_point_apple[1] + distancia[1]]
                                #worksheet.cell(row=row_index, column=4, value=distance)
                                #print("Distance between center point of calice and origin: ", distance_c)
                                #cv2.line(im0, (center_point_apple[0], center_point_apple[1]), (int(center_point_c[0]),int(center_point_c[1])), (23,204,146), 5)

                                #---------------------------------------Angle operations--------------------------------------------------#
                                #Compute current angle of the calice relative to the x axis of the center point of apple plane
                                if distance_c > 0:
                                    calice_angle = math.acos(
                                        round(((tlx_c + brx_c) / 2) -
                                              ((tlx_a + brx_a) / 2)) / distance_c) * (180 / math.pi)
                                    #print("Previous angle of calice: ", round(prev_calice_angle,1))
                                    print('Current angle of calice: ', round(calice_angle, 1))

                                #Compute the rotation by subtracting the current angle to the previous one
                                if prev_calice_angle == None:
                                    diff_calice_angle = 0
                                else:
                                    diff_calice_angle = prev_calice_angle - calice_angle

                                #Print results
                                if diff_calice_angle > 0:
                                    print('Apple rotated: ', round(diff_calice_angle, 1), ' degrees about z-axis (cw)')
                                elif diff_calice_angle < 0:
                                    print('Apple rotated: ', round(diff_calice_angle, 1), ' degrees in z-axis (ccw)')
                                else:
                                    print("Apple didn't rotate about z-axis")

                                prev_calice_angle = calice_angle

                                #---------------------------------------Movement operations--------------------------------------------------#
                                #Compute translation in x-axis of the detected pedunculo
                                if prev_center_point_c_x == None:
                                    translation_c_x = 0
                                else:
                                    translation_c_x = prev_center_point_c_x - current_center_point_c_x

                                #Print results
                                if translation_c_x < 0:
                                    print('Apple rotated', round(translation_c_x, 1), 'degrees about y-axis (ccw)')
                                elif translation_c_x > 0:
                                    print('Calice moved', round(translation_c_x, 1), 'degrees to left (cw)')
                                elif translation_c_x == 0:
                                    print("Apple didn't rotate about y-axis")

                                #Compute translation in y-axis of the detected pedunculo
                                if prev_center_point_c_y == None:
                                    translation_c_y = 0
                                else:
                                    translation_c_y = prev_center_point_c_y - current_center_point_c_y

                                #Print results
                                if translation_c_y < 0:
                                    print('Apple rotated', round(translation_c_y, 1), 'degrees about x-axis (cw)')
                                elif translation_c_y > 0:
                                    print('Apple rotated', round(translation_c_y, 1), 'degrees about x-axis (ccw)')
                                elif translation_c_y == 0:
                                    print("Apple didn't rotate about the x-axis")

                                #Update the previous coordinates of the center point of the cale with the current ones
                                prev_center_point_c_x = current_center_point_c_x
                                prev_center_point_c_y = current_center_point_c_y

                                #---------------------------------------Rotate apple Calice---------------------------------------------#
                                if center_point_apple[0] != 0 and center_point_apple[1] != 0:
                                    try:
                                        if rotation is None or rotation == 0:
                                            print(diff_calice_angle, 'diff pedunculo angle')
                                            rotation = diff_calice_angle

                                        object_line = np.array(
                                            [[center_point_c[0], int(center_point_c[1])],
                                             [int(distancia_1[0]), int(distancia_1[1])]],
                                            dtype=np.float32)

                                        first_half_apple = np.array(
                                            [[int(brx_a), int(bry_a)], [int(distancia_1[0]),
                                                                        int(distancia_1[1])]],
                                            dtype=np.float32)

                                        second_half_apple = np.array([[int(
                                            distancia_1[0]), int(distancia_1[1])], [int(brx_a), int(bry_a)]],
                                                                     dtype=np.float32)

                                        all_apple = np.array(
                                            [[int(tlx_a), int(tly_a)], [int(brx_a), int(bry_a)]], dtype=np.float32)

                                        cv2.line(im0, (int(object_line[0, 0]), int(object_line[0, 1])),
                                                 (int(object_line[1, 0]), int(object_line[1, 1])), (100, 100, 100), 10)

                                        # Define the vertices of the polygon to be filled
                                        vertices_second_half = np.array([[
                                            int(object_line[0, 0]),
                                            int(object_line[0, 1])], [int(object_line[1, 0]),
                                                                      int(object_line[1, 1])], [int(brx_a),
                                                                                                int(tly_a)],
                                                                         [int(brx_a), int(bry_a)]])

                                        vertices_first_half = np.array([[
                                            int(object_line[0, 0]),
                                            int(object_line[0, 1])], [int(object_line[1, 0]),
                                                                      int(object_line[1, 1])], [int(tlx_a),
                                                                                                int(tly_a)],
                                                                        [int(tlx_a), int(bry_a)]])

                                        calice_angle_y = math.acos(
                                            round(((tly_c + bry_c) / 2) -
                                                  ((tly_a + bry_a) / 2)) / distance_c) * (180 / math.pi)

                                        calice_angle_x = math.acos(
                                            round(((tlx_c + brx_c) / 2) -
                                                  ((tlx_a + brx_a) / 2)) / distance_c) * (180 / math.pi)

                                        if center_point_apple[0] < center_point_c[0]:
                                            calice_angle_y = calice_angle_y * (-1)

                                        #print(pedunculo_angle_x ,"angulo X")
                                        #print(pedunculo_angle_y, "angulo Y")

                                        #Calculate the rotation axis and angle
                                        dx = object_line[1, 0] - object_line[0, 0]
                                        dy = object_line[1, 1] - object_line[0, 1]

                                        axis = np.array([dx, dy, 0])

                                        # Normalize the axis
                                        axis = axis / np.linalg.norm(axis)

                                        # Calculate the rotation matrix using Rodrigues' rotation formula
                                        theta = diff_calice_angle
                                        #print(theta ,"theta")
                                        cos_theta = np.cos(theta)
                                        sin_theta = np.sin(theta)

                                        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                                                      [-axis[1], axis[0], 0]])

                                        R = np.eye(3) * cos_theta + (1 - cos_theta) * np.outer(axis,
                                                                                               axis) + sin_theta * K

                                        # Calculate the rotation matrix using the rotation angle
                                        #R = cv2.getRotationMatrix2D(center=(center_point_p[0], center_point_p[1]), angle=theta, scale=1.0)

                                        object_line = np.hstack([object_line, np.ones((2, 1))])
                                        object_line = np.dot(R, object_line.T).T[:, :2]

                                        first_half_apple = np.hstack([first_half_apple, np.ones((2, 1))])
                                        first_half_apple = np.dot(R, first_half_apple.T).T[:, :2]

                                        second_half_apple = np.hstack([second_half_apple, np.ones((2, 1))])
                                        second_half_apple = np.dot(R, second_half_apple.T).T[:, :2]

                                        all_apple = np.hstack([all_apple, np.ones((2, 1))])

                                        #cv2.rectangle(im0, (int(all_apple[0,0]), int(all_apple[0,1])),(int(all_apple[1,0]) , int(all_apple[1,1])), (100, 100, 100), 10)

                                        # Calculate the rotation matrix to rotate around the center of the apple
                                        M = cv2.getRotationMatrix2D((135, 135), round(calice_angle_y, 1), 1.0)
                                        all_apple = np.dot(M, all_apple.T).T[:, :2]

                                        # Calculate the dimensions of the rotated image

                                        rotated_rect = cv2.minAreaRect(all_apple.astype(np.int32))
                                        rotated_box = cv2.boxPoints(rotated_rect)
                                        rotated_box = np.int0(rotated_box)
                                        w, h = np.int0(np.max(rotated_box, axis=0) - np.min(rotated_box, axis=0))

                                        # Calculate the scaling factor
                                        scale_factor = max(w, h) / 320

                                        # Apply the rotation matrix to the cropped image
                                        rotated_img = cv2.warpAffine(cropped_img, M, (w, h))

                                        # Resize the rotated image to fill the 320x320 image
                                        resized_img = cv2.resize(rotated_img,
                                                                 (int(w / scale_factor), int(h / scale_factor)))

                                        # Calculate the translation vector
                                        tx = (320 - resized_img.shape[1]) // 2
                                        ty = (320 - resized_img.shape[0]) // 2

                                        # Apply the translation to the rotated and resized image
                                        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                                        final_img = cv2.warpAffine(resized_img, translation_matrix, (320, 320))

                                        # Show the final image
                                        #cv2.imshow("Final Image", final_img)
                                        #cv2.waitKey(0)

                                        #cv2.imshow("Original Image",cropped_img)
                                        #cv2.waitKey(0)

                                        cropped_img = cv2.warpAffine(cropped_img, M, (320, 320))
                                        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                                        filename1 = f'matches1-{timestamp}.jpg'
                                        cv2.imwrite(f'/home/gabriel/projeto/yolov5/runs/detect/calice/{filename1}',
                                                    cropped_img)
                                        #cv2.imshow("Rotated Image Calice",cropped_img)
                                        #cv2.waitKey(0)

                                    except Exception as e:
                                        traceback.print_exc()
                                        print(f'Error: {e}')

                                if cropped_img is not None:
                                    result.paste(Image.fromarray(cropped_img), (whidth, height))
                                    #result.show()
                                    height += 0
                                    whidth += (w)

                            #------------------------------------------------IF DETECTED PEDUNCULO----------------------------------------------------#
                            if cls == 5 and n_calice == 0 and n_pedunculo == 0:

                                print('\n', s)
                                print('\033[1mPEDUNCULO DETECTED\033[0m')

                                #store the coordinate points of the bounding box of the detected class "pedunculo" in a queue
                                cpoints.put(c1[0])
                                cpoints.put(c1[1])
                                cpoints.put(c2[0])
                                cpoints.put(c2[1])

                                #gets the coordinate points of the bounding box of the detected class "pedunculo" from the queue and passes to variables
                                tlx_p = cpoints.get()
                                tly_p = cpoints.get()
                                brx_p = cpoints.get()
                                bry_p = cpoints.get()

                                #Compute center point of apple plane detection
                                longitude = bry_a - tly_a
                                latitude = brx_a - tlx_a
                                diff_LoLa = longitude - latitude
                                #Compute center point of pedunculo detection
                                center_point_p = round((tlx_p + brx_p) / 2), round((tly_p + bry_p) / 2)
                                center_point_apple = round((tlx_a + brx_a) / 2), round((tly_a + bry_a) / 2)
                                if diff_LoLa > 75:
                                    center_point_apple = round((tlx_a + brx_a) / 2), round((tly_a + bry_a) / 2)
                                    if center_point_apple[0] > center_point_p[0]:
                                        a = (center_point_apple[0] - center_point_p[0] / 2)
                                        center_point_apple = round(center_point_p[0] - a), round((tly_a + bry_a) / 2)
                                    else:
                                        a = ((center_point_p[0] - center_point_apple[0]) / 2)
                                        center_point_apple = round(center_point_p[0] + a), round((tly_a + bry_a) / 2)

                                #print("diff pedunculo X", diff_pedunculo_x)
                                diff_pedunculo_y = longitude - (bry_a - center_point_p[1])
                                diff_pedunculo_x = latitude - (brx_a - center_point_p[0])

                                apple1 = 0
                                if diff_pedunculo_x < latitude * 0.2 or diff_pedunculo_y < longitude * 0.2:
                                    print('teste')
                                    apple1 += 1

                                print('latitude', latitude * 0.20)
                                print('diff pedunculo X', diff_pedunculo_x)
                                #print("diff pedunculo Y", diff_pedunculo_y)
                                #print("Center point of apple", center_point_apple)
                                cv2.circle(im0, center_point_apple, 5, (255, 0, 0), 2)

                                #-----------------PEDUNCULO Operations--------------------------------#

                                current_center_point_p_x = (tlx_p + brx_p) / 2
                                current_center_point_p_y = (tly_p + bry_p) / 2

                                #-----------------APPLE Operations--------------------------------#

                                #Compute distance between center point of apple plane and center point of pedunculo defection
                                distance_p = math.dist((center_point_apple[0], center_point_apple[1]),
                                                       (center_point_p[0], center_point_p[1]))
                                #worksheet.cell(row=row_index, column=4, value=distance)
                                #print("Distance between center point of pedunculo and origin: ", distance_p)
                                print('Distance Pedunculo', distance_p)

                                distancia = [
                                    center_point_apple[0] - center_point_p[0],
                                    center_point_apple[1] - center_point_p[1]]
                                distancia_1 = [
                                    center_point_apple[0] + distancia[0], center_point_apple[1] + distancia[1]]

                                #cv2.line(im0, (int(center_point_apple[0]), int(center_point_apple[1])), (int(center_point_p[0]),int(center_point_p[1])), (72,249,10), 5)
                                #cv2.line(im0, (int(center_point_apple[0]), int(center_point_apple[1])), (int(distancia_1[0]),int(distancia_1[1])), (72,249,10), 5)

                                #cv2.line(im0, (int(center_point_p[0]),int(center_point_p[1])), (int(distancia_1[0]),int(distancia_1[1])), (72,249,10), 5)

                                # Calculate the new endpoint of the line rotated 90 degrees counterclockwise
                                line_dir = np.array([
                                    center_point_p[0] - center_point_apple[0],
                                    center_point_p[1] - center_point_apple[1]])

                                # Rotate the direction vector by 90 degrees counterclockwise
                                rotated_dir = np.array([-line_dir[1], line_dir[0]])

                                # Calculate the endpoint of the second line

                                second_line_end = np.array([center_point_apple[0], center_point_apple[1]]) + rotated_dir

                                # Draw the second line
                                #cv2.line(im0, (center_point_apple[0], center_point_apple[1]), (int(second_line_end[0]), int(second_line_end[1])), (100, 100, 100), 10)

                                center_x_apple = (brx_a + tlx_a)
                                center_y_apple = (bry_a + tly_a)
                                if center_x_apple != 0 and center_y_apple != 0:
                                    center_x_apple = center_x_apple / 2
                                    center_y_apple = center_y_apple / 2

                                    #cv2.line(im0, (int(center_x_apple), tly_a), (int(center_x_apple), bry_a), (50,50,50), 2)

                                #---------------------------------------Angle operations--------------------------------------------------#
                                #
                                #Compute current angle of the pedunculo relative to the x axis of the center point of the apple plane
                                pedunculo_angle = 0
                                if distance_p > 0:
                                    try:
                                        pedunculo_angle = math.acos(
                                            round(((tlx_p + brx_p) / 2) -
                                                  ((tlx_a + brx_a) / 2)) / distance_p) * (180 / math.pi)
                                        if prev_pedunculo_angle == None:
                                            print('Previous angle of pedunculo: 0 degrees')
                                        else:
                                            print('Previous angle of pedunculo: ', round(prev_pedunculo_angle, 1))
                                        print('Current angle of pedunculo: ', round(pedunculo_angle, 1))
                                    except Exception as e:
                                        traceback.print_exc()
                                        print(f'Error: {e}')

                                #Compute the rotation by subtracting the current angle to the previous one
                                print(pedunculo_angle, 'pedunculo_angle')
                                if prev_pedunculo_angle == None:
                                    diff_pedunculo_angle = 0
                                else:
                                    diff_pedunculo_angle = prev_pedunculo_angle - pedunculo_angle

                                #Print results
                                if diff_pedunculo_angle > 0:
                                    print('Apple rotated: ', round(diff_pedunculo_angle, 1),
                                          ' degrees about z-axis (cw)')
                                elif diff_pedunculo_angle < 0:
                                    print('Apple rotated: ', round(diff_pedunculo_angle, 1),
                                          ' degrees about z-axis (ccw)')
                                else:
                                    print("Apple didn't rotate about z-axis")

                                #Update previous angle of pedunculo in relation to the center point of the apple plane with the current one
                                prev_pedunculo_angle = pedunculo_angle

                                #---------------------------------------Movement operations--------------------------------------------------#
                                #Compute translation in x-axis of the detected pedunculo
                                if prev_center_point_p_x == None:
                                    translation_p_x = 0
                                else:
                                    translation_p_x = prev_center_point_p_x - current_center_point_p_x

                                #Print results
                                if translation_p_x < 0:
                                    print('Apple rotated', round(translation_p_x, 1), 'degrees about y-axis (ccw)')
                                elif translation_p_x > 0:
                                    print('Apple rotated', round(translation_p_x, 1), 'degrees about y-axis (cw)')
                                elif translation_p_x == 0:
                                    print("Apple didn't rotate about y-axis")

                                #Compute translation in y-axis of the detected pedunculo
                                if prev_center_point_p_y == None:
                                    translation_p_y = 0
                                else:
                                    translation_p_y = prev_center_point_p_y - current_center_point_p_y

                                #Print results
                                if translation_p_y < 0:

                                    print('Apple rotated', round(translation_p_y, 1), 'degrees about x-axis (cw)')
                                    #print("Ponto X", prev_center_point_p_x , "Ponto X")
                                    #print("Ponto Y", prev_center_point_p_y , "Ponto Y")
                                elif translation_p_y > 0:
                                    print('Apple rotated', round(translation_p_y, 1), 'degrees about x-axis (ccw)')
                                elif translation_p_y == 0:
                                    print("Apple didn't rotate about x-axis")

                                #Update the previous coordinates of the center point of the pedunculo with the current ones
                                prev_center_point_p_x = current_center_point_p_x
                                prev_center_point_p_y = current_center_point_p_y

                                #---------------------------------------Rotate apple--------------------------------------------------#
                                if center_x_apple != 0 and center_y_apple != 0:
                                    try:
                                        if rotation is None or rotation == 0:
                                            print(diff_pedunculo_angle, 'diff pedunculo angle')
                                            rotation = diff_pedunculo_angle

                                        object_line = np.array(
                                            [[center_point_p[0], int(center_point_p[1])],
                                             [int(distancia_1[0]), int(distancia_1[1])]],
                                            dtype=np.float32)

                                        first_half_apple = np.array(
                                            [[int(brx_a), int(bry_a)], [int(distancia_1[0]),
                                                                        int(distancia_1[1])]],
                                            dtype=np.float32)

                                        second_half_apple = np.array([[int(
                                            distancia_1[0]), int(distancia_1[1])], [int(brx_a), int(bry_a)]],
                                                                     dtype=np.float32)

                                        all_apple = np.array(
                                            [[int(tlx_a), int(tly_a)], [int(brx_a), int(bry_a)]], dtype=np.float32)

                                        #cv2.line(im0, (int(object_line[0,0]), int(object_line[0,1])),(int(object_line[1,0]) , int(object_line[1,1])), (100, 100, 100), 10)
                                        #cv2.rectangle(im0, (int(tlx_a), int(tly_a)),(int(brx_a) , int(bry_a)), (255, 255, 255), 10)

                                        # Define the vertices of the polygon to be filled
                                        vertices_second_half = np.array([[
                                            int(object_line[0, 0]),
                                            int(object_line[0, 1])], [int(object_line[1, 0]),
                                                                      int(object_line[1, 1])], [int(brx_a),
                                                                                                int(tly_a)],
                                                                         [int(brx_a), int(bry_a)]])

                                        vertices_first_half = np.array([[
                                            int(object_line[0, 0]),
                                            int(object_line[0, 1])], [int(object_line[1, 0]),
                                                                      int(object_line[1, 1])], [int(tlx_a),
                                                                                                int(tly_a)],
                                                                        [int(tlx_a), int(bry_a)]])

                                        #pedunculo_angle_x = math.acos(round((((tlx_p+brx_p)/2)-((tlx_a+brx_a)/2)))/distance_p) * (180/math.pi)
                                        #pedunculo_angle_y = math.acos(round((((tly_p+bry_p)/2)-((tly_a+bry_a)/2)))/distance_p) * (180/math.pi)
                                        pedunculo_angle_y = math.acos(
                                            round(((tly_a + bry_a) / 2) -
                                                  ((tly_p + bry_p) / 2)) / distance_p) * (180 / math.pi)
                                        #print(pedunculo_angle_y,"pedunculo Angle Y")
                                        #pedunculo_angle_x = math.acos(round((((tlx_p+brx_p)/2)-((tlx_a+brx_a)/2)))/distance_p) * (180/math.pi)
                                        #print(distance_p, "distance_p")
                                        if center_point_apple[0] > center_point_p[0]:
                                            pedunculo_angle_y = pedunculo_angle_y * (-1)

                                        #print(pedunculo_angle ,"angulo X")
                                        #print(pedunculo_angle_y, "angulo Y")

                                        #Calculate the rotation axis and angle
                                        dx = object_line[1, 0] - object_line[0, 0]
                                        dy = object_line[1, 1] - object_line[0, 1]

                                        # Create an empty mask with the same size as the original image

                                        img1 = np.zeros((320, 320, 3), dtype=np.uint8)
                                        mask = np.zeros_like(im0)

                                        #if rotation > 0 :
                                        #print("Maior")

                                        # Fill the polygon with white color
                                        #cv2.fillPoly(mask, [vertices_first_half], (255, 255, 255))

                                        #kernel = np.ones((320, 320), np.uint8)
                                        #dilated_mask = cv2.dilate(mask, kernel, iterations=1)

                                        # Extract the region of interest from the original image
                                        #masked_img = cv2.bitwise_and(im0, mask)

                                        #roi = cv2.bitwise_and(im0, dilated_mask)

                                        # Show the image
                                        #cv2.imshow('Original Image', masked_img)
                                        #cv2.waitKey(0)
                                        #cv2.destroyAllWindows()

                                        # Create an empty image with the same size as the original image
                                        #im0_filled = np.zeros_like(im0)

                                        # Fill the polygon with a specified color and thickness
                                        #cv2.fillPoly(im0_filled, [vertices_first_half], (100, 100, 100))

                                        # Combine the filled polygon with the original image
                                        #im0 = cv2.addWeighted(im0, 1, im0_filled, 0.5, 0)
                                        #print(second_half_apple)
                                        #cv2.imshow("Cropped Image",im0)
                                        #cv2.waitKey(0)
                                        #print(diff_pedunculo_angle)

                                        #diff_pedunculo_angle = diff_pedunculo_angle + 150

                                        #angle = round(translation_p_x, 1)
                                        #print(angle , "angle")
                                        #else:
                                        # Fill the polygon with white color
                                        #cv2.fillPoly(mask, [vertices_second_half], (255, 255, 255))

                                        # Extract the region of interest from the original image
                                        #masked_img = cv2.bitwise_and(im0, mask)
                                        # Show the image
                                        #cv2.imshow('Original Image', masked_img)
                                        #cv2.waitKey(0)
                                        #cv2.destroyAllWindows()

                                        axis = np.array([dx, dy, 0])

                                        # Normalize the axis
                                        axis = axis / np.linalg.norm(axis)

                                        # Calculate the rotation matrix using Rodrigues' rotation formula
                                        theta = diff_pedunculo_angle
                                        #print(theta ,"theta")
                                        cos_theta = np.cos(theta)
                                        sin_theta = np.sin(theta)

                                        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                                                      [-axis[1], axis[0], 0]])

                                        R = np.eye(3) * cos_theta + (1 - cos_theta) * np.outer(axis,
                                                                                               axis) + sin_theta * K

                                        # Calculate the rotation matrix using the rotation angle
                                        #R = cv2.getRotationMatrix2D(center=(center_point_p[0], center_point_p[1]), angle=theta, scale=1.0)

                                        # Calculate the rotation matrix using the rotation angle
                                        #R = cv2.getRotationMatrix2D(center=(center_point_p[0], center_point_p[1]), angle=theta, scale=1.0)

                                        object_line = np.hstack([object_line, np.ones((2, 1))])
                                        object_line = np.dot(R, object_line.T).T[:, :2]

                                        first_half_apple = np.hstack([first_half_apple, np.ones((2, 1))])
                                        first_half_apple = np.dot(R, first_half_apple.T).T[:, :2]

                                        second_half_apple = np.hstack([second_half_apple, np.ones((2, 1))])
                                        second_half_apple = np.dot(R, second_half_apple.T).T[:, :2]

                                        all_apple = np.hstack([all_apple, np.ones((2, 1))])
                                        all_apple = np.dot(R, all_apple.T).T[:, :2]
                                        #cv2.line(im0, (int(center_point_p[0]),int(center_point_p[1])), (int(distancia_1[0]),int(distancia_1[1])), (72,249,10), 5)
                                        #cv2.rectangle(im0, (int(all_apple[0,0]), int(all_apple[0,1])),(int(all_apple[1,0]) , int(all_apple[1,1])), (100, 100, 100), 10)

                                        #rotated_object_corners = np.dot(R, all_apple.T).T[:, :2]

                                        #center_point = tuple(np.mean(tuple(object_line), axis=0))
                                        # Calculate the rotation matrix to rotate around the center of the apple
                                        M = cv2.getRotationMatrix2D((135, 135), round(pedunculo_angle_y, 1), 1.0)

                                        # Apply the rotation matrix to the image
                                        # Calculate the dimensions of the rotated image

                                        rotated_rect = cv2.minAreaRect(all_apple.astype(np.int32))
                                        rotated_box = cv2.boxPoints(rotated_rect)
                                        rotated_box = np.int0(rotated_box)
                                        w, h = np.int0(np.max(rotated_box, axis=0) - np.min(rotated_box, axis=0))

                                        # Calculate the scaling factor
                                        scale_factor = max(w, h) / 320

                                        # Apply the rotation matrix to the cropped image
                                        cropped_img = cv2.warpAffine(cropped_img, M, (350, 350))

                                        # Resize the rotated image to fill the 320x320 image
                                        resized_img = cv2.resize(cropped_img,
                                                                 (int(300 / scale_factor), int(3000 / scale_factor)))

                                        # Calculate the translation vector
                                        tx = (340 - resized_img.shape[1]) // 2
                                        ty = (340 - resized_img.shape[0]) // 2

                                        # Apply the translation to the rotated and resized image
                                        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                                        final_img = cv2.warpAffine(resized_img, translation_matrix, (400, 400))

                                        #cv2.rectangle(cropped_img, (int(all_apple[0,0]), int(all_apple[0,1])),(int(all_apple[1,0]) , int(all_apple[1,1])), (100, 100, 100), 10)

                                        x, y, w, h = cv2.boundingRect(all_apple.astype(np.int32))

                                        cropped_img_rot = cropped_img[y:y + h, x:x + w]

                                        #cv2.imshow("Cropped img rot",cropped_img_rot)
                                        #cv2.waitKey(0)
                                        # Show the final image
                                        #cv2.imshow("Rotated Image", cropped_img)
                                        #cv2.waitKey(0)

                                        # Show the final image
                                        #cv2.imshow("Final Image", final_img)
                                        #cv2.waitKey(0)

                                        #cv2.imshow("Original Image",cropped_img)
                                        #cv2.waitKey(0)

                                        x, y, w, h = cv2.boundingRect(all_apple.astype(np.int32))

                                        cropped_img_rot = cropped_img[y:y + h, x:x + w]

                                        #cv2.imshow("Cropped img rot",cropped_img_rot)
                                        #cv2.waitKey(0)
                                        result = sr.upsample(cropped_img)

                                        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

                                        filename1 = f'matches1-{timestamp}.jpg'

                                        if apple1 > 0:
                                            #cv2.imwrite(f"/home/gabriel/projeto/yolov5/runs/detect/pedunculo/1/{filename1}", cropped_img)
                                            cv2.imwrite(
                                                f'/home/gabriel/projeto/yolov5/runs/detect/pedunculo/1/{filename1}',
                                                result)

                                        #else :
                                        #cv2.imwrite(f"/home/gabriel/projeto/yolov5/runs/detect/pedunculo/2/{filename1}", cropped_img)

                                        if prev_img is not None:
                                            # Calculate homography matrix between fixed points
                                            #fixed_pts1 = np.float32([[0,0], [0,200], [200,0], [200,200]])
                                            #fixed_pts2 = np.float32([[0,0],[200,0],[0,200],[200,200]])
                                            #homography_matrix, _ = cv2.findHomography(fixed_pts1, fixed_pts2)

                                            #cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                                            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

                                            # Remove as bordas da imagem
                                            #border_size = 70
                                            #cropped_img = cropped_img[border_size:-border_size, border_size:-border_size]
                                            #prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

                                            # Apply Gaussian blur to the grayscale images
                                            #cropped_img = cv2.GaussianBlur(cropped_img, (5, 5), cv2.BORDER_DEFAULT)

                                            # Here, we'll use the SimpleBlobDetector algorithm to detect circular blobs in the image
                                            params = cv2.SimpleBlobDetector_Params()
                                            params.filterByArea = True
                                            params.minArea = 1
                                            params.maxArea = 320
                                            detector = cv2.SimpleBlobDetector_create(params)

                                            # Detect blobs in the first image
                                            keypoints1 = detector.detect(prev_img)

                                            # Detect blobs in the second image
                                            keypoints2 = detector.detect(result)

                                            # Create SIFT descriptor extractor

                                            edge_threshold = int(1 * min(320, 320))
                                            sift = cv2.SIFT_create(nfeatures=1000,
                                                                   contrastThreshold=0.023,
                                                                   sigma=0.9,
                                                                   edgeThreshold=edge_threshold,
                                                                   nOctaveLayers=9)

                                            # Compute descriptors for keypoints
                                            kp1, des1 = sift.compute(prev_img, keypoints1)
                                            kp2, des2 = sift.compute(result, keypoints2)

                                            imagem1 = cv2.drawKeypoints(
                                                prev_img,
                                                kp1,
                                                None,
                                                color=(0, 255, 0),
                                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                            #cv2.imshow("Image Kp",imagem1)
                                            #cv2.waitKey(0)
                                            # Create matcher object
                                            matcher = cv2.FlannBasedMatcher(
                                                dict(algorithm=255,
                                                     table_number=6,
                                                     key_size=12,
                                                     crossCheck=True,
                                                     multi_probe_lever=1))
                                            #matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)

                                            matches = matcher.knnMatch(des1, des2, k=2)

                                            good_matches = []
                                            for m, n in matches:
                                                if m.distance < 0.9 * n.distance:
                                                    if len(good_matches) == 0:
                                                        good_matches.append(m)
                                                    else:
                                                        # Apply Lowe's test
                                                        match_distances = [m.distance for m in good_matches]
                                                        min_distance = min(match_distances)
                                                        if m.distance < 1.2 * min_distance:
                                                            good_matches.append(m)

                                            matches = good_matches
                                            matches = sorted(matches, key=lambda x: x.distance)

                                            # Here, we'll use the homography transformation to map the image points to a common coordinate system
                                            src_pts = np.float32([kp1[m.queryIdx].pt
                                                                  for m in matches]).reshape(-1, 1, 2)
                                            dst_pts = np.float32([kp2[m.trainIdx].pt
                                                                  for m in matches]).reshape(-1, 1, 2)
                                            #src_pts = cv2.perspectiveTransform(src_pts, homography_matrix)
                                            #print(src_pts , "Source")
                                            #print(dst_pts , "Destine")
                                            # Estima a homografia entre as imagens
                                            try:
                                                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                                                #print(M, "Homography matrix")
                                                if M is None or M.shape != (3, 3):
                                                    raise ValueError('Homography matrix is not 3x3')

                                                # check the type of the matrix
                                                if M.dtype != np.float32 and M.dtype != np.float64:
                                                    raise ValueError('Homography matrix has invalid type')

                                                # Calcula a largura e a altura da imagem resultante
                                                width = prev_img.shape[1] + result.shape[1]
                                                height = prev_img.shape[0] + result.shape[0]

                                                # Une as imagens usando a homografia
                                                res = cv2.warpPerspective(prev_img, M, (width, height))
                                                stem_points = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), M)

                                                # Step 4: Use the transformed stem coordinates to create a 2D model of the apple
                                                # Here, we'll fit an ellipse to the transformed stem coordinates using the OpenCV fitEllipse function
                                                ellipse = cv2.fitEllipse(stem_points)
                                                # Display the results
                                                result = cv2.ellipse(prev_img, ellipse, (0, 255, 0), 2)
                                                #cv2.imshow("Ellipse",res)
                                                #cv2.waitKey(0)
                                                # Draw the corresponding matches in two images
                                                timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                                                imMatches = cv2.drawMatches(
                                                    prev_img,
                                                    kp1,
                                                    result,
                                                    kp2,
                                                    matches,
                                                    None,
                                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                                filenameM1 = f'matchesM1-{timestamp}.jpg'
                                                cv2.imwrite(
                                                    f'/home/gabriel/projeto/yolov5/runs/detect/matches/{filenameM1}',
                                                    imMatches)
                                                prev_img = result
                                                #prev_img = cropped_img
                                            except cv2.error as e:
                                                timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                                                imMatches = cv2.drawMatches(
                                                    prev_img,
                                                    kp1,
                                                    result,
                                                    kp2,
                                                    matches,
                                                    None,
                                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                                filenameM1 = f'matchesM1-{timestamp}.jpg'
                                                cv2.imwrite(
                                                    f'/home/gabriel/projeto/yolov5/runs/detect/matches/{filenameM1}',
                                                    imMatches)
                                                prev_img = result
                                                #prev_img = cropped_img
                                                print(f'Error: {e}')

                                        else:
                                            #border_size = 40
                                            #cropped_img = cropped_img[border_size:-border_size, border_size:-border_size]
                                            #prev_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

                                            #result = result[border_size:-border_size, border_size:-border_size]
                                            prev_img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                                            #prev_img = cv2.GaussianBlur(prev_img, (5, 5), cv2.BORDER_DEFAULT)

                                        # Calculate the translation matrices to center the object line at the origin
                                        #M1 = np.array([[1, 0, -center_point[0]], [0, 1, -center_point[1]]])
                                        #M2 = np.array([[1, 0, center_point[0]], [0, 1, center_point[1]]])
                                        # Crop the image

                                        #if cropped_img_heigth is None:
                                        #    cropped_img_heigth = h

                                        #cropped_img1 = im0[50,50]
                                        # Display the cropped image
                                        #cv2.waitKey(0)
                                        #cv2.destroyAllWindows()

                                    except Exception as e:
                                        traceback.print_exc()
                                        print(f'Error: {e}')

                                # Calculate the rotation matrix
                                #center = tuple(np.mean(tuple(object_corners), axis=0))
                                #center_point = tuple(np.mean(tuple(object_line), axis=0))

                                #M = cv2.getRotationMatrix2D(center_point_apple, round(pedunculo_angle_y ,1), 1.0)

                                # Calculate the translation matrices
                                #M1 = np.array([[1, 0, -center_point[0]], [0, 1, -center_point[1]], [0, 0, 1]])
                                #M2 = np.array([[1, 0, center_point[0]], [0, 1, center_point[1]], [0, 0, 1]])

                                # Calculate the rotation matrix to rotate around the line
                                #M3 = cv2.getRotationMatrix2D((0, 0), angle, 1.0)

                                # Apply the rotation to the object corners
                                #rotated_object_corners1 = cv2.transform(np.array([object_line]), M)

                                # Calculate the bounding box of the rotated object
                                #x1, y1, w1, h1 = cv2.boundingRect(rotated_object_corners1.astype(np.int32))
                                #cropped_img1 = im0[y1:y1 + h1, x1:x1+w1]
                                #cv2.imshow("Cropped Image",cropped_img1)
                                #cv2.waitKey(0)

                                # Apply the rotation to the object line and calculate the bounding box of the rotated object

                                # Create RGBA images with alpha channel set by the masks
                                #img1_rgba = cv2.cvtColor(imc, cv2.COLOR_BGR2RGBA)
                                #_, mask = cv2.threshold(cv2.cvtColor(imc, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY)
                                #img1_rgba[translation_p_y:,translation_p_x :,diff_pedunculo_angle] = mask

                                if cropped_img is not None:
                                    #result.paste(Image.fromarray(cropped_img), (whidth, height))
                                    #result.show()
                                    height += 0
                                    whidth += (w)

                        #----------------------------------END OF OPERATIONS---------------------------------------#

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':

                    #result.save(f"{save_path}{im0}")
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    result.save(f'{save_path}')
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
