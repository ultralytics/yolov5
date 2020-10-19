import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

import string
import json
import numpy as np
import datetime

RESIZE_FACTOR = 2


def id_generator(size=4, chars=string.digits + string.ascii_uppercase, nb=None):
    if nb is None:
        generated_id = "".join(random.choice(chars) for _ in range(size))
    else:
        generated_id = f"{convert_to_base(chars, nb):0>{size}}"
    return generated_id


def convert_to_base(chars, n):
    base = len(chars)
    if n < base:
        return chars[n]
    else:
        return convert_to_base(chars, n // base) + chars[n % base]


def json_dump_wrap(data, out_path, indent=2):
    """
    Verify the type in data to make sure they are JSON-serializable
    """
    data = get_serializable_structure(data)

    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Dump data
    with open(out_path, "w") as outfile:
        json.dump(data, outfile, indent=indent)


def get_serializable_structure(data):
    # Construct non JSON serializable object types
    invalid_int_value = list(
        map(type, [np.int8(16), np.int16(16), np.int32(16), np.int64(16)])
    )
    invalid_uint_value = list(
        map(type, [np.uint8(16), np.uint16(16), np.uint32(16), np.uint64(16)])
    )
    invalid_float_value = list(map(type, [np.float16(16), np.float32(16)]))
    datetime_type = type(datetime.datetime.now())
    path_type = type(Path())
    none_values = [np.nan, np.inf]

    # Types which requires to look every object stored in them
    list_type = type([])  # For-loop
    array_type = type(np.array([]))
    tuple_type = type(())
    dict_type = type({})  # reccursive call

    # Initialize function to check "1D" objects
    def get_serializable_atomic_value(value):
        if type(value) == datetime_type:
            return (
                value.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            )  # Convert to iso with timezone
        elif type(value) in (invalid_int_value + invalid_uint_value):
            return int(value)
        elif type(value) in invalid_float_value:
            return np.float64(value)
        elif type(value) == path_type:
            return str(value)
        elif value in none_values:
            return None
        else:
            return value

    def _get_serializable_structure(structure):
        if type(structure) is array_type:
            return [_get_serializable_structure(value) for value in structure.flatten()]
        elif type(structure) is list_type:
            return [_get_serializable_structure(value) for value in structure]
        elif type(structure) is tuple_type:
            return tuple(_get_serializable_structure(value) for value in structure)
        elif type(structure) is dict_type:
            # Verify types in data
            for key in structure.keys():
                structure[key] = _get_serializable_structure(structure[key])
            return structure
        else:
            return get_serializable_atomic_value(structure)

    # Verify types in data
    data = _get_serializable_structure(data)
    return data


def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    save_txt = False
    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = False
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    centroids = []
    current_centroid_id = 0
    current_count = 0
    for path, img, im0s, vid_cap in dataset:
        if current_count != dataset.count:
            video_key = Path(dataset.files[current_count]).resolve()
            out_path = video_key.parent / (video_key.stem + "_YOLO_output") / "centroids_with_meta.json"
            centroids_with_meta = {
                "centroids": centroids,
                "extra_information": {"resize_factor": RESIZE_FACTOR},
            }
            json_dump_wrap(centroids_with_meta, out_path, indent=2)

            centroids = []
            current_count = dataset.count

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            centroids_in_frame = []
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for bbox in xyxy2xywh(det[:, :4]).tolist():
                    centroid_bbox = [
                        max((bbox[0]-bbox[2]/2)//RESIZE_FACTOR, 0),
                        max((bbox[1]-bbox[3]/2)//RESIZE_FACTOR, 0),
                        bbox[2]//RESIZE_FACTOR,
                        bbox[3]//RESIZE_FACTOR
                    ]
                    centroids_in_frame.append(
                        {
                            "bbox": centroid_bbox,
                            "center": [
                                round(bbox[0]/RESIZE_FACTOR, 2),
                                round(bbox[1]/RESIZE_FACTOR, 2),
                            ],
                            "class": 1,
                            "interpolated": False,
                            "id": id_generator(nb=current_centroid_id),
                        }
                    )
                    current_centroid_id += 1

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            centroids.append(centroids_in_frame)

            # Print time (inference + NMS)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    video_key = Path(dataset.files[current_count]).resolve()
    out_path = video_key.parent / (video_key.stem + "_YOLO_output") / "centroids_with_meta.json"
    centroids_with_meta = {
        "centroids": centroids,
        "extra_information": {"resize_factor": RESIZE_FACTOR},
    }
    json_dump_wrap(centroids_with_meta, out_path, indent=2)

    centroids = []
    current_count = dataset.count

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt)


if __name__ == '__main__':
    import sys
    import d6tflow
    import luigi

    d6tflow.settings.log_level = "ERROR"
    karolinska_capture = Path("../../data/Karolinska/capture_MAST_data")

    class TaskDetectKarolinska(d6tflow.tasks.TaskPickle):
        date = luigi.Parameter()
        patient_id = luigi.Parameter()

        def run(self):
            if not "--source" in sys.argv:
                sys.argv += ["--source", "To be replaced"]
            source = karolinska_capture / self.date / self.patient_id
            if list(source.glob("*.avi")):
                sys.argv[sys.argv.index("--source") + 1] = source.resolve().as_posix()
                main()
            self.save(source)


    class TaskTrackKarolinska(d6tflow.tasks.TaskPickle):
        date = luigi.Parameter()
        patient_id = luigi.Parameter()

        def run(self):
            for path in list((karolinska_capture / date / patient_id).glob("*.avi")):
                analyze_frames(
                    path,
                    run_detection=False,
                    run_classification=True,
                    run_tracking=True,
                    run_viz=True,
                )

    if "--localization" in sys.argv:
        for date in os.listdir(karolinska_capture):
            for patient_id in os.listdir(karolinska_capture / date):
                task = TaskDetectKarolinska(date=date, patient_id=patient_id)
                d6tflow.preview(task)
                d6tflow.run(task)

    if "--tracking" in sys.argv:
        from nanovare_casa_core.analysis.analysis import analyze_frames
        for date in os.listdir(karolinska_capture):
            for patient_id in os.listdir(karolinska_capture / date):
                task = TaskTrackKarolinska(date=date, patient_id=patient_id)
                d6tflow.preview(task)
                d6tflow.run(task)
