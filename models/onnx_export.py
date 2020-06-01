import argparse

import onnx

from models.common import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='../weights/yolov5s.pt', help='model path RELATIVE to ./models/')
    parser.add_argument('--img-size', default=640, help='inference size (pixels)')
    parser.add_argument('--batch-size', default=1, help='batch size')
    opt = parser.parse_args()

    # Parameters
    f = opt.weights.replace('.pt', '.onnx')  # onnx filename
    img = torch.zeros((opt.batch_size, 3, opt.img_size, opt.img_size))  # image size, (1, 3, 320, 192) iDetection

    # Load pytorch model
    google_utils.attempt_download(opt.weights)
    model = torch.load(opt.weights)['model']
    model.eval()
    # model.fuse()  # optionally fuse Conv2d + BatchNorm2d layers TODO

    # Export to onnx
    model.model[-1].export = True  # set Detect() layer export=True
    torch.onnx.export(model, img, f, verbose=False, opset_version=11)

    # Check onnx model
    model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model)  # check onnx model
    print(onnx.helper.printable_graph(model.graph))  # print a human readable representation of the graph
    print('Export complete. ONNX model saved to %s\nView with https://github.com/lutzroeder/netron' % f)
