# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
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
import os
import platform
import sys
from pathlib import Path

import torch

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
# 这些都是用户自定义的库，由于上一步已经把路径加载上了，所以现在可以导入，这个顺序不可以调换。
#
# 用的时候再解释这些库/方法的作用
# run方法 48~213行
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image# 一张图片上检测的最大目标数量
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results# 是否在推理时预览图片
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3# 过滤指定类的预测结果
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)# 绘制Bounding_box的线宽度
        hide_labels=False,  # hide labels# True: 隐藏置信度
        hide_conf=False,  # hide confidences# True: 隐藏置信度
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source) # 是否需要保存图片,如果nosave(传入的参数)为false且source的结尾不是txt则保存图片
    # 后面这个source.endswith('.txt')也就是source以.txt结尾，不过我不清楚这是什么用法
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # 而IMG_FORMATS 和 VID_FORMATS两个变量保存的是所有的视频和图片的格式后缀
    # 判断source是不是视频/图像文件路径
    # 假如source是"D://YOLOv5/data/1.jpg"，则Path(source).suffix是".jpg",Path(source).suffix[1:]是"jpg"

    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))  # 判断source是否是链接
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)  # 判断是source是否是摄像头
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download  # 如果source是一个指向图片/视频的链接,则下载输入数据

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run  # save_dir是保存运行结果的文件夹名，是通过递增的方式来命名的。
    # 第一次运行时路径是“runs\detect\exp”，第二次运行时路径是“runs\detect\exp1”
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir  # 根据前面生成的路径创建文件夹

    # Load model
    device = select_device(device)# select_device方法定义在utils.torch_utils模块中，返回值是torch.device对象，
    # 也就是推理时所使用的硬件资源。输入值如果是数字，表示GPU序号。也可是输入‘cpu’，表示使用CPU训练，默认是cpu
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # DetectMultiBackend定义在models.common模块中，是我们要加载的网络，其中weights参数就是输入时指定的权重文件（比如yolov5s.pt）
    stride, names, pt = model.stride, model.names, model.pt
    # stride：推理时所用到的步长，默认为32， 大步长适合于大目标，小步长适合于小目标
    # names：保存推理结果名的列表，比如默认模型的值是['person', 'bicycle', 'car', ...]
    # pt: 加载的是否是pytorch模型（也就是pt格式的文件），

    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # 将图片大小调整为步长的整数倍
    # 比如假如步长是10，imagesz是[100,101],则返回值是[100,100]

    # Dataloader
    bs = 1  # batch_size
    if webcam:  # 使用摄像头作为输入
        view_img = check_imshow(warn=True)  # 检测cv2.imshow()方法是否可以执行，不能执行则抛出异常
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # 加载输入数据流
        # source：输入数据源 image_size 图片识别前被放缩的大小， stride：识别时的步长，
        # auto的作用可以看utils.augmentations.letterbox方法，它决定了是否需要将图片填充为正方形，如果auto=True则不需要

        bs = len(dataset)  # batch_size 批大小
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs  # 用于保存视频,前者是视频路径,后者是一个cv2.VideoWriter对象

    # Run inference开始预测
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup  # 使用空白图片（零矩阵）预先用GPU跑一遍预测流程，可以加速预测
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # seen: 已经处理完了多少帧图片
    # windows: 如果需要预览图片,windows列表会给每个输入文件存储一个路径.
    # dt: 存储每一步骤的耗时

    for path, im, im0s, vid_cap, s in dataset:
        # 在dataset中，每次迭代的返回值是self.sources, img, img0, None, ''
        # path：文件路径（即source）
        # im: 处理后的输入图片列表（经过了放缩操作）
        # im0s: 源输入图片列表
        # vid_cap
        # s： 图片的基本信息，比如路径，大小

        with dt[0]:  # 获取当前时间
            im = torch.from_numpy(im).to(model.device)  #将图片放到指定设备(如GPU)上识别
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32  # 把输入从整型转化为半精度/全精度浮点数。
            im /= 255  # 0 - 255 to 0.0 - 1.0 #将图片归一化处理（这是图像表示方法的的规范，使用浮点数就要归一化）
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim 添加一个第0维。在pytorch的nn.Module的输入中，第0维是batch的大小，这里添加一个1。

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 如果为True则保留推理过程中的特征图，保存在runs文件夹中
            pred = model(im, augment=augment, visualize=visualize) # 推理结果，pred保存的是所有的bound_box的信息，


        #  NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # 执行非极大值抑制，返回值为过滤后的预测框
            # conf_thres： 置信度阈值
            # iou_thres： iou阈值
            # classes: 需要过滤的类（数字列表）
            # agnostic_nms： 标记class-agnostic或者使用class-specific方式。默认为class-agnostic
            # max_det: 检测框结果的最大数量

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image  # 每次迭代处理一张图片，
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                # frame：此次取的是第几张图片
                s += f'{i}: '  # s后面拼接一个字符串i
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg  # 推理结果图片保存的路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string  # 显示推理前裁剪后的图像尺寸
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh#得到原图的宽和高
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 如果save_crop的值为true， 则将检测到的bounding_box单独保存成一张图片。
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # 得到一个绘图的类，类中预先存储了原图、线条宽度、类名

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # 将标注的bounding_box大小调整为和原图一致（因为训练时原图经过了放缩）

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # 打印出所有的预测结果  比如1 person（检测出一个人）

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # 将坐标转变成x y w h 的形式，并归一化
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        # line的形式是： ”类别 x y w h“，假如save_conf为true，则line的形式是：”类别 x y w h 置信度“
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            # 写入对应的文件夹里，路径默认为“runs\detect\exp*\labels”

                    if save_img or save_crop or view_img:  # Add bbox to image# 给图片添加推理后的bounding_box边框
                        c = int(cls)  # integer class# 类别标号
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # 绘制边框
                    if save_crop:  # 将预测框内的图片单独保存
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            # im0是绘制好的图片
            if view_img:  # 如果view_img为true,则显示该图片
                if platform.system() == 'Linux' and p not in windows:
                    # 如果当前图片/视频的路径不在windows列表里,则说明需要重新为该图片/视频创建一个预览窗口
                    windows.append(p)  # 标记当前图片/视频已经创建好预览窗口了
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0) # 预览图片
                cv2.waitKey(1)  # 1 millisecond  # 暂停 1 millisecond

            # Save results (image with detections)
            if save_img:  # 如果save_img为true,则保存绘制完的图片
                if dataset.mode == 'image':# 如果是图片,则保存
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'# 如果是视频或者"流"
                    if vid_path[i] != save_path:  # new video
                        # vid_path[i] != save_path,说明这张图片属于一段新的视频,需要重新创建视频文件
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
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")  # 打印耗时

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image平均每张图片所耗费时间
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''# 标签保存的路径
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
