"""Exports a pytorch *.pt model to *.torchscript format

Usage:
    $ export PYTHONPATH="$PWD" && python models/torchscript_export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse


from models.common import *
from utils import google_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    print(opt)

    # Parameters
    f = opt.weights.replace('.pt', '.torchscript')  # onnx filename
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size, (1, 3, 320, 192) iDetection

    # Load pytorch model
    google_utils.attempt_download(opt.weights)
    model = torch.load(opt.weights, map_location=torch.device('cpu'))['model'].float()
    model.eval()

    # Don't fuse layers, it won't work with torchscript exports
    #model.fuse()

    # Export to jit/torchscript
    model.model[-1].export = True  # set Detect() layer export=True
    _ = model(img)  # dry run

    traced_script_module = torch.jit.trace(model, img)
    traced_script_module.save(f)