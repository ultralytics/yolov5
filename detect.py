import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from numpy import random
import numpy as np

from models.experimental import attempt_load
from models.yolo import Detect
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    if len(imgsz) == 1:
        imgsz = imgsz[0]

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    with open('data/coco.yaml') as f:
        names = yaml.load(f, Loader=yaml.FullLoader)['names']  # class names (assume COCO)

    if weights[0].split('.')[-1] == 'pt':
        backend = 'pytorch'
        names = model.module.names if hasattr(model, 'module') else model.names  # class names
    elif weights[0].split('.')[-1] == 'pb':
        backend = 'graph_def'
    elif weights[0].split('.')[-1] == 'tflite':
        backend = 'tflite'
    else:
        backend = 'saved_model'

    if backend == 'saved_model' or backend =='graph_def' or backend=='tflite':
       import tensorflow as tf
       from tensorflow import keras

    if backend == 'pytorch':
        model = attempt_load(weights, map_location=device)  # load FP32 model
    elif backend == 'saved_model':
        if tf.__version__.startswith('1'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            sess = tf.Session(config=config)
            loaded = tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING], weights[0])
            tf_input = loaded.signature_def['serving_default'].inputs['input_1']
            if not opt.no_tf_nms:
                tf_output = loaded.signature_def['serving_default'].outputs['tf__detect']
            else:
                tf_outputs = [loaded.signature_def['serving_default'].outputs['tf_op_layer_CombinedNonMaxSuppression'],
                              loaded.signature_def['serving_default'].outputs['tf_op_layer_CombinedNonMaxSuppression_1'],
                              loaded.signature_def['serving_default'].outputs['tf_op_layer_CombinedNonMaxSuppression_2'],
                              loaded.signature_def['serving_default'].outputs['tf_op_layer_CombinedNonMaxSuppression_3']
                ]
        else:
            model = keras.models.load_model(weights[0])
    elif backend == 'graph_def':
        if tf.__version__.startswith('1'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            sess = tf.Session(config=config)
            graph = tf.Graph()
            graph_def = graph.as_graph_def()
            graph_def.ParseFromString(open(weights[0], 'rb').read())
            tf.import_graph_def(graph_def, name='')
            default_graph = tf.get_default_graph()
            tf_input = default_graph.get_tensor_by_name('x:0')
            if not opt.no_tf_nms:
                tf_output = default_graph.get_tensor_by_name('Identity:0')
            else:
                tf_outputs = [default_graph.get_tensor_by_name('Identity:0'),
                              default_graph.get_tensor_by_name('Identity_1:0'),
                              default_graph.get_tensor_by_name('Identity_2:0'),
                              default_graph.get_tensor_by_name('Identity_3:0')
                ]

        else:
            # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            # https://github.com/leimao/Frozen_Graph_TensorFlow
            def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
                def _imports_graph_def():
                    tf.compat.v1.import_graph_def(graph_def, name="")

                wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
                import_graph = wrapped_import.graph

                if print_graph == True:
                    print("-" * 50)
                    print("Frozen model layers: ")
                    layers = [op.name for op in import_graph.get_operations()]
                    for layer in layers:
                        print(layer)
                    print("-" * 50)

                return wrapped_import.prune(
                    tf.nest.map_structure(import_graph.as_graph_element, inputs),
                    tf.nest.map_structure(import_graph.as_graph_element, outputs))

            graph = tf.Graph()
            graph_def = graph.as_graph_def()
            graph_def.ParseFromString(open(weights[0], 'rb').read())
            frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                            inputs="x:0",
                                            outputs="Identity:0" if not opt.no_tf_nms else
                                                ["Identity:0", "Identity_1:0", "Identity_2:0", "Identity_3:0"],
                                            print_graph=False)

    elif backend == 'tflite':
        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(
            model_path=opt.weights[0],
            experimental_delegates=
                [tf.lite.experimental.load_delegate('libedgetpu.so.1')] if opt.edgetpu else None)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    if backend == 'pytorch':
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    if half and backend == 'pytorch':
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, auto=backend == 'pytorch')
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, auto=backend == 'pytorch')

    # Get names and colors
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    img = torch.zeros((1, 3, *imgsz), device=device)  # init img
    if (backend == 'saved_model' or backend == 'graph_def') and tf.__version__.startswith('1'):
        fetches = tf_output.name if not opt.no_tf_nms else [o.name for o in tf_outputs]

    if backend == 'pytorch':
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    elif backend == 'saved_model':
        if tf.__version__.startswith('1'):
            _ = sess.run(fetches, feed_dict={tf_input.name: img.permute(0, 2, 3, 1).cpu().numpy()})
        else:
            _ = model(img.permute(0, 2, 3, 1).cpu().numpy(), training=False)
    elif backend == 'graph_def':
        if tf.__version__.startswith('1'):
            _ = sess.run(fetches, feed_dict={tf_input.name: img.permute(0, 2, 3, 1).cpu().numpy()})
        else:
            _ = frozen_func(x=tf.constant(img.permute(0, 2, 3, 1).cpu().numpy()))
    elif backend == 'tflite':
        input_data = img.permute(0, 2, 3, 1).cpu().numpy()
        if opt.tfl_int8:
            input_data = input_data.astype(np.uint8)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half and backend == 'pytorch' else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        if backend == 'pytorch':
            pred = model(img, augment=opt.augment)[0]

        elif backend == 'saved_model':
            if tf.__version__.startswith('1'):
                pred = sess.run(fetches, feed_dict={tf_input.name: img.permute(0, 2, 3, 1).cpu().numpy()})
                if not opt.no_tf_nms:
                    pred = torch.tensor(pred)
            else:
                res = model(img.permute(0, 2, 3, 1).cpu().numpy(), training=False)
                if not opt.no_tf_nms:
                    pred = res[0].numpy()
                    pred = torch.tensor(pred)
                else:
                    pred = res[0]

        elif backend == 'graph_def':
            if tf.__version__.startswith('1'):
                pred = sess.run(fetches, feed_dict={tf_input.name: img.permute(0, 2, 3, 1).cpu().numpy()})
                if not opt.no_tf_nms:
                    pred = torch.tensor(pred)
            else:
                pred = frozen_func(x=tf.constant(img.permute(0, 2, 3, 1).cpu().numpy()))
                if not opt.no_tf_nms:
                    pred = torch.tensor(pred.numpy())

        elif backend == 'tflite':
            input_data = img.permute(0, 2, 3, 1).cpu().numpy()
            if opt.tfl_int8:
                scale, zero_point = input_details[0]['quantization']
                input_data = input_data / scale + zero_point
                input_data = input_data.astype(np.uint8)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            if not opt.tfl_detect:
                output_data = interpreter.get_tensor(output_details[0]['index'])
                pred = torch.tensor(output_data)
            else:
                yaml_file = Path(opt.cfg).name
                with open(opt.cfg) as f:
                    cfg = yaml.load(f, Loader=yaml.FullLoader)

                anchors = cfg['anchors']
                nc = cfg['nc']
                nl = len(anchors)
                x = [torch.tensor(interpreter.get_tensor(output_details[i]['index']), device=device) for i in range(nl)]
                if opt.tfl_int8:
                    for i in range(nl):
                        scale, zero_point = output_details[i]['quantization']
                        x[i] = x[i].float()
                        x[i] = (x[i] - zero_point) * scale

                def _make_grid(nx=20, ny=20):
                    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
                    return torch.stack((xv, yv), 2).view((1, 1, ny * nx, 2)).float()

                no = nc + 5
                grid = [torch.zeros(1)] * nl  # init grid
                a = torch.tensor(anchors).float().view(nl, -1, 2).to(device)
                anchor_grid = a.clone().view(nl, 1, -1, 1, 2)  # shape(nl,1,na,1,2)
                z = []  # inference output
                for i in range(nl):
                    _, _, ny_nx, _ = x[i].shape
                    r = imgsz[0] / imgsz[1]
                    nx = int(np.sqrt(ny_nx / r))
                    ny = int(r * nx)
                    grid[i] = _make_grid(nx, ny).to(x[i].device)
                    stride = imgsz[0] // ny
                    y = x[i].sigmoid()
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i].to(x[i].device)) * stride  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
                    z.append(y.view(-1, no))

                pred = torch.unsqueeze(torch.cat(z, 0), 0)

        # Apply NMS
        if not opt.no_tf_nms:
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        else:
            nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = pred
            if not tf.__version__.startswith('1'):
                nmsed_boxes = torch.tensor(nmsed_boxes.numpy())
                nmsed_scores = torch.tensor(nmsed_scores.numpy())
                nmsed_classes = torch.tensor(nmsed_classes.numpy())
                valid_detections = torch.tensor(valid_detections.numpy())
            else:
                nmsed_boxes = torch.tensor(nmsed_boxes)
                nmsed_scores = torch.tensor(nmsed_scores)
                nmsed_classes = torch.tensor(nmsed_classes)
                valid_detections = torch.tensor(valid_detections)
            bs = nmsed_boxes.shape[0]
            pred = [None] * bs
            for i in range(bs):
                pred[i] = torch.cat([nmsed_boxes[i, :valid_detections[i], :],
                                     torch.unsqueeze(nmsed_scores[i, :valid_detections[i]], -1),
                                     torch.unsqueeze(nmsed_classes[i, :valid_detections[i]], -1)], -1)

        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
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

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', nargs='+', type=int, default=[640], help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
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
    parser.add_argument('--tfl-detect', action='store_true', help='add Detect module in TFLite')
    parser.add_argument('--cfg', type=str, default='./models/yolov5s.yaml', help='cfg path')
    parser.add_argument('--tfl-int8', action='store_true', help='use int8 quantized TFLite model')
    parser.add_argument('--no-tf-nms', action='store_true', help='dont proceed NMS due to model w/ TensorFlow NMS')
    parser.add_argument('--edgetpu', action='store_true', help='inference with Edge TPU')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
