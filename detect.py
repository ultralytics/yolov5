# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py
        --source path/to/img.jpg
        --weights yolov5s.pt
        --img 640
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, \
    check_requirements, check_suffix, colorstr, increment_path, \
    non_max_suppression, print_args, save_one_box, scale_coords, \
    set_logging, strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
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
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    source, save_img, webcam, save_dir = initialize_fields(
        source, nosave, project, name, exist_ok, save_txt
    )

    device = initialize_device(device, half)
    
    w, classify, pt, onnx, tflite, pb, saved_model = \
        check_model_type(weights)

    model, modelc, net, session, frozen_func, interpreter, \
    int8, input_details, output_details, imgsz = \
        load_model(
            weights, imgsz, device, half, dnn,
            w, classify, pt, onnx, tflite, pb, saved_model
        )

    dataset, bs, vid_path, vid_writer = initialize_dataloader(
        webcam, source, imgsz, stride, pt
    )

    dt, seen = run_inference(half, device, imgsz, dataset, model,
        net, session, frozen_func, interpreter, img, pt, onnx, dnn,
        tflite, int8, input_details, output_details, save_dir,
        path, visualize, augment, conf_thres, iou_thres, classes,
        agnostic_nms, max_det, classify, modelc, webcam,
        save_crop, save_txt, save_conf, save_img, hide_labels, hide_conf,
        view_img, video_path, txt_path, video_cap, line_thickness, names
    )

    print_results(
        dt, seen, imgsz, save_dir, save_txt, save_img, update, weights
    )


def initialize_fields(source, nosave, project, name, exist_ok, save_txt):
    source = str(source)
    
    # save inference images
    save_img = not nosave and not source.endswith('.txt')
    webcam = \
        source.isnumeric() or \
        source.endswith('.txt') or \
        source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://')
        )

    # Directories
    ## increment run
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)

    ## make dir
    (save_dir / 'labels' if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  

    return source, save_img, webcam, save_dir


def initialize_device(device, half):
    set_logging()
    device = select_device(device)
    
    # half precision only supported on CUDA
    half &= device.type != 'cpu'

    return device


def check_model_type(weights):
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = \
        False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    
    # check weights have acceptable suffix
    check_suffix(w, suffixes)
    
    # backend booleans
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)

    return w, classify, pt, onnx, tflite, pb, saved_model


def load_model(weights, imgsz, device, half, dnn, w, classify, pt,
        onnx, tflite, pb, saved_model):
    # assign defaults
    stride, names = 64, [f'class{i}' for i in range(1000)]
    model = modelc = net = session = frozen_func = interpreter = None
    if pt:
        model, stride, modelc = load_model_pre_trained(
            w, weights, device, half, classify
        )
    elif onnx:
        net, session = load_model_onnx(dnn, w)
    else:  # TensorFlow models
        frozen_func, model, interpreter, int8, \
        input_details, output_details = load_model_tensorflow(
            w, pb, saved_model, tflite
        )
    
    # check image size
    imgsz = check_img_size(imgsz, s=stride)

    return model, modelc, net, session, frozen_func, interpreter, \
        int8, input_details, output_details, imgsz


def load_model_pre_trained(w, weights, device, half, classify):
    model = \
        torch.jit.load(w) if 'torchscript' in w \
        else attempt_load(weights, map_location=device)
    
    # model stride
    stride = int(model.stride.max())
    
    # get class names
    names = model.module.names if hasattr(model, 'module') else model.names
    if half:
        # to FP16
        model.half()

    # second-stage classifier
    modelc = None
    if classify:
        # initialize
        modelc = load_classifier(name='resnet50', n=2)
        modelc.load_state_dict(
            torch.load(
                'resnet50.pt', map_location=device
            )['model']).to(device).eval()

    return model, stride, modelc


def load_model_onnx(dnn, w):
    net = session = None
    if dnn:
        # check_requirements(('opencv-python>=4.5.4',))
        net = cv2.dnn.readNetFromONNX(w)
    else:
        check_requirements(
            ('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime')
        )
        
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)

    return net, session


def load_model_tensorflow(w, pb, saved_model, tflite):
    check_requirements(('tensorflow>=2.4.1',))

    frozen_func = model = interpreter = None
    
    import tensorflow as tf
    if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
        def wrap_frozen_graph(gd, inputs, outputs):
            # wrapped import
            x = tf.compat.v1.wrap_function(
                lambda: tf.compat.v1.import_graph_def(gd, name=""), []
            )
            return x.prune(
                tf.nest.map_structure(x.graph.as_graph_element, inputs),
                tf.nest.map_structure(x.graph.as_graph_element, outputs)
            )

        graph_def = tf.Graph().as_graph_def()
        graph_def.ParseFromString(open(w, 'rb').read())
        frozen_func = wrap_frozen_graph(
            gd=graph_def, inputs="x:0", outputs="Identity:0"
        )
    elif saved_model:
        model = tf.keras.models.load_model(w)
    elif tflite:
        # load TFLite model
        interpreter = tf.lite.Interpreter(model_path=w)
        # allocate
        interpreter.allocate_tensors()
        # inputs
        input_details = interpreter.get_input_details()
        # outputs
        output_details = interpreter.get_output_details()
        # is TFLite quantized uint8 model
        int8 = input_details[0]['dtype'] == np.uint8

    return frozen_func, model, interpreter, \
        int8, input_details, output_details


def initialize_dataloader(webcam, source, imgsz, stride, pt):
    bs = None
    dataset = None
    if webcam:
        view_img = check_imshow()
        # set True to speed up constant image size inference
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    
    vid_path, vid_writer = [None] * bs, [None] * bs

    return dataset, bs, vid_path, vid_writer


def run_inference(half, device, imgsz, dataset, model, \
        net, session, frozen_func, interpreter, img, pt, onnx, dnn, \
        tflite, int8, input_details, output_details, save_dir, \
        path, visualize, augment, conf_thres, iou_thres, classes, \
        agnostic_nms, max_det, classify, modelc, webcam, \
        save_crop, save_txt, save_conf, save_img, hide_labels, hide_conf, \
        view_img, video_path, txt_path, video_cap, line_thickness, names):
    if pt and device.type != 'cpu':
        # run once
        model(
            torch.zeros(1, 3, *imgsz).to(
                device
            ).type_as(next(model.parameters()))
        )
    
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()
        img = pre_process(onnx, img, device, half)
        t2 = time_sync()
        dt[0] += t2 - t1
        pred = inference(
            model, net, session, frozen_func, interpreter, img, pt, \
            onnx, dnn, tflite, int8, input_details, output_details, save_dir, \
            path, visualize, augment
        )
        t3 = time_sync()
        dt[1] += t3 - t2
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes,
            agnostic_nms, max_det=max_det
        )
        dt[2] += time_sync() - t3

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        seen += process_predictions(
            pred, webcam, img, im0s, dataset, save_dir,
            save_crop, save_txt, save_conf, save_img, hide_labels, hide_conf,
            view_img, video_path, txt_path, video_cap, line_thickness, names,
            t3, t2
        )

    return dt, seen


def pre_process(onnx, img, device, half):
    if onnx:
        img = img.astype('float32')
    else:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    return img


def inference(model, net, session, frozen_func, interpreter, img, pt, \
        onnx, dnn, tflite, int8, input_details, output_details, save_dir, \
        path, visualize, augment):
    pred = None
    if pt:
        visualize = increment_path(
            save_dir / Path(path).stem, mkdir=True
        ) if visualize else False
        pred = model(img, augment=augment, visualize=visualize)[0]
    elif onnx:
        if dnn:
            net.setInput(img)
            pred = torch.tensor(net.forward())
        else:
            pred = torch.tensor(
                session.run(
                    [session.get_outputs()[0].name],
                    {session.get_inputs()[0].name: img}
                )
            )
    else:  # tensorflow model (tflite, pb, saved_model)
        # image in numpy
        imn = img.permute(0, 2, 3, 1).cpu().numpy()
        if pb:
            pred = frozen_func(x=tf.constant(imn)).numpy()
        elif saved_model:
            pred = model(imn, training=False).numpy()
        elif tflite:
            if int8:
                scale, zero_point = input_details[0]['quantization']
                # de-scale
                imn = (imn / scale + zero_point).astype(np.uint8)
            
            interpreter.set_tensor(input_details[0]['index'], imn)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])
            if int8:
                scale, zero_point = output_details[0]['quantization']
                # re-scale
                pred = (pred.astype(np.float32) - zero_point) * scale
        
        pred[..., 0] *= imgsz[1]  # x
        pred[..., 1] *= imgsz[0]  # y
        pred[..., 2] *= imgsz[1]  # w
        pred[..., 3] *= imgsz[0]  # h
        pred = torch.tensor(pred)

    

    return pred


def process_predictions(pred, webcam, img, im0s, dataset, save_dir, \
        save_crop, save_txt, save_conf, save_img, hide_labels, hide_conf, \
        view_img, video_path, txt_path, video_cap, line_thickness, names, \
        t3, t2):
    seen = 0
    
    # per image
    for i, det in enumerate(pred):
        seen += 1
        p = s = im0 = frame = None
        if webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
        else:
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

        save_path, text_path, s, gn, imc, annotator = \
            initialize_prediction_fields(
                p, save_dir, dataset, frame, img,
                im0, save_crop, line_thickness, names
            )
        if len(det):
            write_results(
                det, img, im0, imc, annotator, names, save_txt,
                save_conf, hide_labels, hide_conf, txt_path, save_img,
                save_crop, view_img, gn
            )
        
        print_inference_time(s, t3, t2)

        im0 = annotator.result()
        if view_img:
            view_stream_results()

        if save_img:
            save_image_with_annotation(
                dataset, save_path, im0, video_path, i, video_cap
            )

    return seen

def initialize_prediction_fields(p, save_dir, dataset, frame, img, \
        im0, save_crop, line_thickness, names):
    # to Path
    p = Path(p)
    
    # img.jpg
    save_path = str(save_dir / p.name)
    
    # img.txt
    txt_path = str(
        save_dir / 'labels' / p.stem
    ) + (
        '' if dataset.mode == 'image' else f'_{frame}'
    )
    
    # print string
    s += '%gx%g ' % img.shape[2:]

    # normalization gain whwh
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

    # for save_crop
    imc = im0.copy() if save_crop else im0
    
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))

    return save_path, text_path, s, gn, imc, annotator


def write_results(det, img, im0, imc, annotator, names, save_txt, \
        save_conf, hide_labels, hide_conf, txt_path, save_img, \
        save_crop, view_img, gn):
    p = Path(p)

    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(
        img.shape[2:], det[:, :4], im0.shape
    ).round()

    # Print results
    for c in det[:, -1].unique():
        # detections per class
        n = (det[:, -1] == c).sum()
        # add to string
        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

    # Write results
    for *xyxy, conf, cls in reversed(det):
        # Write to file
        if save_txt:
            # normalized xywh
            xywh = (
                xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn
            ).view(-1).tolist()
            # label format
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
            with open(txt_path + '.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

        # Add bbox to image
        if save_img or save_crop or view_img:
            c = int(cls)  # integer class
            label = None if hide_labels else (
                names[c] if hide_conf else f'{names[c]} {conf:.2f}'
            )
            annotator.box_label(xyxy, label, color=colors(c, True))
            if save_crop:
                save_one_box(
                    xyxy, imc,
                    file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                    BGR=True
                )


def print_inference_time(s, t3, t2):
    print(f'{s}Done. ({t3 - t2:.3f}s)')


def view_stream_results():
    cv2.imshow(str(p), im0)
    
    # 1 millisecond
    cv2.waitKey(1)


def save_image_with_annotation(dataset, save_path, im0, video_path, index, \
        video_cap):
    if dataset.mode == 'image':
        cv2.imwrite(save_path, im0)
    else:  # 'video' or 'stream'
        # new video
        if vid_path[index] != save_path:
            vid_path[index] = save_path
            if isinstance(vid_writer[index], cv2.VideoWriter):
                # release previous video writer
                vid_writer[index].release()
            
            # video
            fps = w = h = None
            if vid_cap:
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # stream
                fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path += '.mp4'
            
            vid_writer[index] = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
            )
        
        vid_writer[index].write(im0)


# TODO: Break long lines here
def print_results(dt, seen, imgsz, save_dir, save_txt,
        save_img, update, weights):
    # speeds per image
    t = tuple(x / seen * 1E3 for x in dt)
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
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
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
