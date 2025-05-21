# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Convert JSON format annotated file to standard TXT that supported by YOLOV5 framework.

Usage:
    $ python json2txt.py --data custom-dataset.yaml
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import LOGGER, check_dataset, check_file, print_args


def parse_opt(known=False):
    """
    Parse command-line arguments for annotated file format converting.

    Args:
        known (bool, optional): If True, parses known arguments, ignoring the unknown. Defaults to False.

    Returns:
        (argparse.Namespace): Parsed command-line arguments containing options for YOLOv5 execution.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/custom-dataset.yaml", help="dataset.yaml path")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    """
    Runs the main entry point for converting.

    Args:
        opt (argparse.Namespace): The command-line arguments parsed.

    Returns:
        None
    """
    print_args(vars(opt))
    opt.data = check_file(opt.data)  # str
    data_dict = check_dataset(opt.data)
    # Create label:idx map
    data_dict["idxs"] = {v: k for k, v in data_dict["names"].items()}
    idxs = data_dict["idxs"]
    train_path = data_dict["train"]

    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings

    for image_file in glob.glob(str(Path(train_path) / "**" / "*.*"), recursive=True):
        label_file = sb.join(image_file.rsplit(sa, 1)).rsplit(".", 1)[0] + ".json"
        if not os.path.exists(label_file):
            continue
        label_file2 = sb.join(image_file.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"
        with open(label_file, errors="ignore") as fin:
            label_dict = json.load(fin)
            label_dict.pop("imageData")
            img_w, img_h = label_dict["imageWidth"], label_dict["imageHeight"]
            bbox_infos = []
            for i, bbox_dict in enumerate(label_dict["shapes"]):
                cls = bbox_dict["label"]
                if bbox_dict["shape_type"] == "rectangle" and cls in idxs:
                    x1y1, x2y2 = bbox_dict["points"]
                    xl, yt = x1y1  # top-left point
                    xr, yb = x2y2  # bottom-right point
                    if xr < xl:
                        xl, xr = xr, xl
                    if yb < yt:
                        yt, yb = yb, yt
                    LOGGER.info(f"i: {i}, cls: {cls}, xl: {xl}, yt: {yt}, xr: {xr}, yb: {yb}")
                    bbox_w, bbox_h = xr - xl, yb - yt
                    xc, yc = xl + bbox_w / 2, yt + bbox_h / 2
                    line = [idxs[cls], xc / img_w, yc / img_h, bbox_w / img_w, bbox_h / img_h]
                    bbox_infos.append(" ".join(list(map(str, line))))
            with open(label_file2, "w", errors="ignore") as fout:
                fout.write("\n".join(bbox_infos))
    LOGGER.info("Json2Txt done!!!")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
