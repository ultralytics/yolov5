import torch
import numpy as np
from PIL import Image
from utils.augmentations import letterbox
from utils.general import check_img_size, scale_coords
from utils.plots import Annotator, colors


def preprocess_function(input_pre: dict, model, input_parameters: dict) -> dict:
    # Validate image size
    imgsz = input_parameters.get('imgsz', 640)
    stride = int(model.model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    input_parameters['imgsz'] = imgsz
    nparr = np.asarray(input_pre, dtype="uint8")

    # Padded resize
    img = letterbox(nparr, imgsz, stride=stride)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img)
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    return img


# You need to pass a torch.tensor back.
def postprocess_function(pred: torch.tensor, model, input_pre, input_parameters: dict) -> torch.tensor:
    print(pred[0])
    im = preprocess_function(input_pre, model, input_parameters)
    det = pred[0]   # unpack batch
    input_pre = np.ascontiguousarray(input_pre, dtype=np.uint8)
    # input_pre = Image.fromarray(input_pre)
    annotator = Annotator(input_pre, line_width=3)

    # Rescale boxes from img_size to input_pre size
    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], input_pre.shape).round()

    # Write results
    for *xyxy, conf, cls in reversed(det):
        # Add bbox to image
        c = int(cls)  # integer class
        label = f'{model.model.names[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))

    # Output results
    input_pre = annotator.result()
    return input_pre
