# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Run YOLOv5 classification inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python classify/predict.py --weights yolov5s-cls.pt --source 0                               # webcam
                                                                   img.jpg                         # image
                                                                   vid.mp4                         # video
                                                                   screen                          # screenshot
                                                                   path/                           # directory
                                                                   list.txt                        # list of images
                                                                   list.streams                    # list of streams
                                                                   'path/*.jpg'                    # glob
                                                                   'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                                   'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python classify/predict.py --weights yolov5s-cls.pt                 # PyTorch
                                           yolov5s-cls.torchscript        # TorchScript
                                           yolov5s-cls.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                           yolov5s-cls_openvino_model     # OpenVINO
                                           yolov5s-cls.engine             # TensorRT
                                           yolov5s-cls.mlmodel            # CoreML (macOS-only)
                                           yolov5s-cls_saved_model        # TensorFlow SavedModel
                                           yolov5s-cls.pb                 # TensorFlow GraphDef
                                           yolov5s-cls.tflite             # TensorFlow Lite
                                           yolov5s-cls_edgetpu.tflite     # TensorFlow Edge TPU
                                           yolov5s-cls_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator

from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    print_args,
    strip_optimizer,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s-cls.pt",  # model.pt path(s)
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(224, 224),  # inference size (height, width)
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    nosave=False,  # do not save images/videos
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/predict-cls",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    """
    Conducts YOLOv5 classification inference on diverse input sources and saves results.

    Args:
        weights (str | Path): Path(s) to the model weights file(s).
        source (str | Path): File/dir/URL/glob/screen/0(webcam) input source for inference.
        data (str | Path): Path to the dataset configuration file (YAML).
        imgsz (tuple[int, int]): Inference size (height, width).
        device (str): CUDA device, e.g., '0' or '0,1,2,3' or 'cpu'.
        view_img (bool): If True, display results.
        save_txt (bool): If True, save results to *.txt files.
        nosave (bool): If True, do not save inference images/videos.
        augment (bool): If True, apply augmented inference.
        visualize (bool): If True, visualize features.
        update (bool): If True, update all models.
        project (str | Path): Directory to save the results.
        name (str): Name to save the results under `project/name`.
        exist_ok (bool): If True, existing project/name directory is ok; do not increment.
        half (bool): If True, use FP16 half-precision inference.
        dnn (bool): If True, use OpenCV DNN for ONNX inference.
        vid_stride (int): Video frame-rate stride.

    Returns:
        None

    Notes:
        For an exhaustive list of supported input sources and formats along with usage examples, refer to the YOLOv5
        documentation: https://github.com/ultralytics/yolov5#usage

    Examples:
        ```python
        # Run inference on an image
        run(weights='yolov5s-cls.pt', source='img.jpg')

        # Run inference on a video
        run(weights='yolov5s-cls.pt', source='vid.mp4', view_img=True)

        # Run inference from a webcam feed
        run(weights='yolov5s-cls.pt', source=0, view_img=True)
        ```

    DOM import sys
    <|vq_9832|>
    """
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.Tensor(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            results = model(im)

        # Post-process
        with dt[2]:
            pred = F.softmax(results, dim=1)  # probabilities

        # Process predictions
        for i, prob in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt

            s += "%gx%g " % im.shape[2:]  # print string
            annotator = Annotator(im0, example=str(names), pil=True)

            # Print results
            top5i = prob.argsort(0, descending=True)[:5].tolist()  # top 5 indices
            s += f"{', '.join(f'{names[j]} {prob[j]:.2f}' for j in top5i)}, "

            # Write results
            text = "\n".join(f"{prob[j]:.2f} {names[j]}" for j in top5i)
            if save_img or view_img:  # Add bbox to image
                annotator.text([32, 32], text, txt_color=(255, 255, 255))
            if save_txt:  # Write to file
                with open(f"{txt_path}.txt", "a") as f:
                    f.write(text + "\n")

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
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
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """
    Parses command line arguments for YOLOv5 inference settings, allowing configuration of model path, input source,
    device, and various inference options.

    Args:
    weights (str | list[str], optional): Path(s) to model files. Default is 'ROOT / "yolov5s-cls.pt"'.
    source (str, optional): Input source which can be file, directory, URL, glob pattern, or webcam stream. Default is 'ROOT / "data/images"'.
    data (str, optional): Path to dataset.yaml file. Default is 'ROOT / "data/coco128.yaml"'.
    imgsz (list[int], optional): Inference image size specified as height and width. Default is [224].
    device (str, optional): Computation device to be used, e.g., '0' for first GPU, '0,1,2,3' for multi-GPU, or 'cpu' for CPU. Default is ''.
    view_img (bool, optional): Flag to display inference results. Default is False.
    save_txt (bool, optional): Flag to save inference results to a text file. Default is False.
    nosave (bool, optional): Flag to skip saving images/videos from inference results. Default is False.
    augment (bool, optional): Flag for augmented inference. Default is False.
    visualize (bool, optional): Flag to generate and display feature visualizations. Default is False.
    update (bool, optional): Flag to update all models. Default is False.
    project (str, optional): Directory to save the results of inference. Default is 'ROOT / "runs/predict-cls"'.
    name (str, optional): Name of the experiment, appended to the project directory. Default is 'exp'.
    exist_ok (bool, optional): Flag to allow overwriting of existing project/name directory. Default is False.
    half (bool, optional): Flag to use half-precision (FP16) during inference. Default is False.
    dnn (bool, optional): Flag to use OpenCV DNN for ONNX inference. Default is False.
    vid_stride (int, optional): Video frame-rate stride for processing. Default is 1.

    Returns:
    opt (argparse.Namespace): Namespace populated with parsed command line arguments.

    Notes:
    Examples of how to call this function from the command line can be found in the documentation of specific use cases, such as running inference on different types of input sources and formats. For more detailed usage instructions, please refer to https://github.com/ultralytics/ultralytics.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s-cls.pt", help="model path(s)")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[224], help="inference size h,w")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/predict-cls", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes YOLOv5 model inference with options for ONNX DNN and video frame-rate stride adjustments.

    Args:
        opt (argparse.Namespace): The command line arguments for YOLOv5 inference, parsed by `parse_opt()`.
            Attributes include:
            - weights (list[str]): Path to the model weights file(s).
            - source (str): Input source, can be a file, directory, URL, glob, screen, or webcam.
            - data (str): Path to the dataset.yaml file, optional.
            - imgsz (list[int]): Inference size as height, width.
            - device (str): Device to run inference on, specified as '0' or '0,1,2,3' for GPU or 'cpu'.
            - view_img (bool): If True, displays the results using OpenCV.
            - save_txt (bool): If True, saves results to text files.
            - nosave (bool): If True, does not save images or videos.
            - augment (bool): If True, performs augmented inference.
            - visualize (bool): If True, visualizes features.
            - update (bool): If True, updates all models.
            - project (str): Root directory to save results.
            - name (str): Name of the directory to save results.
            - exist_ok (bool): If True, existing project/name is acceptable and will not increment.
            - half (bool): If True, uses FP16 half-precision inference.
            - dnn (bool): If True, uses OpenCV DNN for ONNX inference.
            - vid_stride (int): Frame-rate stride for video processing.

    Returns:
        None

    Note:
        Ensure required packages are installed. Dependencies are listed in `requirements.txt`. To install, run:
        `pip install -r requirements.txt`.

    Example:
        ```python
        opt = parse_opt()
        main(opt)
        ```
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
