"""
Run Gradio demo.
Add --host 0.0.0.0 to share with other machines without --share.

Usage - local model:
python utils/gradio/demo.py --source local --path MODEL_FILE

Usage - github repo:
python utils/gradio/demo.py --source github --model yolov5s
"""

import argparse
import os
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import print_args, check_requirements
from utils.dataloaders import IMG_FORMATS


def predict(inp, conf, iou, agnostic_nms):
    model.conf = conf
    model.iou = iou
    model.agnostic = agnostic_nms
    res = model([inp[..., ::-1]], size=opt.imgsz).render()[0][..., ::-1]
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--share', action='store_true', help='share yolov5 demo with public link')
    parser.add_argument('--host', type=str, default='localhost', help='server ip/name (0.0.0.0 for network request)')
    parser.add_argument('--port', type=int, default=7860, help='server port')
    parser.add_argument('--example_dir', type=str, default=ROOT / 'data/images', help='example image dir')
    parser.add_argument('--source', type=str, default='local', help='torch hub source: github/local')
    parser.add_argument('--model', type=str, default='custom', help='model name used by github source')
    parser.add_argument('--path', type=str, default=ROOT / 'yolov5l.pt', help='local model path')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))

    check_requirements(exclude=('tensorboard', 'thop'))
    check_requirements('gradio')
    import gradio as gr

    files = Path(opt.example_dir).glob('*')
    examples = []
    for f in files:
        if f.suffix.lower()[1:] in IMG_FORMATS:
            examples.append([f])
    kwargs = {'path': opt.path} if opt.source == 'local' else {}
    repo = ROOT if opt.source == 'local' else 'ultralytics/yolov5'  # source == 'github'
    model = torch.hub.load(repo, opt.model, source=opt.source, **kwargs)
    demo = gr.Interface(fn=predict,
                        inputs=[gr.Image(), gr.Slider(0, 1, 0.25), gr.Slider(0, 1, 0.45), gr.Checkbox()],
                        outputs="image",
                        examples=examples)
    demo.launch(share=opt.share, server_name=opt.host, server_port=opt.port)

