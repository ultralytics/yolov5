# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Plotting utils."""

import contextlib
import math
import os
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw
from scipy.ndimage.filters import gaussian_filter1d
from ultralytics.utils.plotting import Annotator

from utils import TryExcept, threaded
from utils.general import LOGGER, clip_boxes, increment_path, xywh2xyxy, xyxy2xywh
from utils.metrics import fitness

# Settings
RANK = int(os.getenv("RANK", -1))
matplotlib.rc("font", **{"size": 11})
matplotlib.use("Agg")  # for writing to files only


class Colors:
    """Provides an RGB color palette derived from Ultralytics color scheme for visualization tasks."""

    def __init__(self):
        """
        Initializes the Colors class with a palette derived from Ultralytics color scheme, converting hex codes to RGB.

        Colors derived from `hex = matplotlib.colors.TABLEAU_COLORS.values()`.
        """
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Returns color from palette by index `i`, in BGR format if `bgr=True`, else RGB; `i` is an integer index."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hexadecimal color `h` to an RGB tuple (PIL-compatible) with order (R, G, B)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results.
    """
    if ("Detect" not in module_type) and (
        "Segment" not in module_type
    ):  # 'Detect' for Object Detect task,'Segment' for Segment task
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis("off")

            LOGGER.info(f"Saving {f}... ({n}/{channels})")
            plt.savefig(f, dpi=300, bbox_inches="tight")
            plt.close()
            np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())  # npy save


def hist2d(x, y, n=100):
    """
    Generates a logarithmic 2D histogram, useful for visualizing label or evolution distributions.

    Used in used in labels.png and evolve.png.
    """
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    """Applies a low-pass Butterworth filter to `data` with specified `cutoff`, `fs`, and `order`."""
    from scipy.signal import butter, filtfilt

    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        """Applies a low-pass Butterworth filter to a signal with specified cutoff frequency, sample rate, and filter
        order.
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype="low", analog=False)

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # forward-backward filter


def output_to_target(output, max_det=300):
    """Converts YOLOv5 model output to [batch_id, class_id, x, y, w, h, conf] format for plotting, limiting detections
    to `max_det`.
    """
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, xyxy2xywh(box), conf), 1))
    return torch.cat(targets, 0).numpy()


@threaded
def plot_images(images, targets, paths=None, fname="images.jpg", names=None):
    """Plots an image grid with labels from YOLOv5 predictions or targets, saving to `fname`."""
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    max_size = 1920  # max image size
    max_subplots = 16  # max image subplots, i.e. 4x4
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        mosaic[y : y + h, x : x + w, :] = im

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text([x + 5, y + 5], text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]  # image targets
            boxes = xywh2xyxy(ti[:, 2:6]).T
            classes = ti[:, 1].astype("int")
            labels = ti.shape[1] == 6  # labels if no conf column
            conf = None if labels else ti[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale < 1:  # absolute coords need scale if image scales
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = f"{cls}" if labels else f"{cls} {conf[j]:.1f}"
                    annotator.box_label(box, label, color=color)
    annotator.im.save(fname)  # save


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=""):
    """Plots learning rate schedule for given optimizer and scheduler, saving plot to `save_dir`."""
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]["lr"])
    plt.plot(y, ".-", label="LR")
    plt.xlabel("epoch")
    plt.ylabel("LR")
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / "LR.png", dpi=200)
    plt.close()


def plot_val_txt():
    """
    Plots 2D and 1D histograms of bounding box centers from 'val.txt' using matplotlib, saving as 'hist2d.png' and
    'hist1d.png'.

    Example: from utils.plots import *; plot_val()
    """
    x = np.loadtxt("val.txt", dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect("equal")
    plt.savefig("hist2d.png", dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig("hist1d.png", dpi=200)


def plot_targets_txt():
    """
    Plots histograms of object detection targets from 'targets.txt', saving the figure as 'targets.jpg'.

    Example: from utils.plots import *; plot_targets_txt()
    """
    x = np.loadtxt("targets.txt", dtype=np.float32).T
    s = ["x targets", "y targets", "width targets", "height targets"]
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label=f"{x[i].mean():.3g} +/- {x[i].std():.3g}")
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig("targets.jpg", dpi=200)


def plot_val_study(file="", dir="", x=None):
    """
    Plots validation study results from 'study*.txt' files in a directory or a specific file, comparing model
    performance and speed.

    Example: from utils.plots import *; plot_val_study()
    """
    save_dir = Path(file).parent if file else Path(dir)
    plot2 = False  # plot additional results
    if plot2:
        ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)[1].ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    # for f in [save_dir / f'study_coco_{x}.txt' for x in ['yolov5n6', 'yolov5s6', 'yolov5m6', 'yolov5l6', 'yolov5x6']]:
    for f in sorted(save_dir.glob("study*.txt")):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        if plot2:
            s = ["P", "R", "mAP@.5", "mAP@.5:.95", "t_preprocess (ms/img)", "t_inference (ms/img)", "t_NMS (ms/img)"]
            for i in range(7):
                ax[i].plot(x, y[i], ".-", linewidth=2, markersize=8)
                ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(
            y[5, 1:j],
            y[3, 1:j] * 1e2,
            ".-",
            linewidth=2,
            markersize=8,
            label=f.stem.replace("study_coco_", "").replace("yolo", "YOLO"),
        )

    ax2.plot(
        1e3 / np.array([209, 140, 97, 58, 35, 18]),
        [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
        "k.-",
        linewidth=2,
        markersize=8,
        alpha=0.25,
        label="EfficientDet",
    )

    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(25, 55)
    ax2.set_xlabel("GPU Speed (ms/img)")
    ax2.set_ylabel("COCO AP val")
    ax2.legend(loc="lower right")
    f = save_dir / "study.png"
    print(f"Saving {f}...")
    plt.savefig(f, dpi=300)


@TryExcept()  # known issue https://github.com/ultralytics/yolov5/issues/5395
def plot_labels(labels, names=(), save_dir=Path("")):
    """Plots dataset labels, saving correlogram and label images, handles classes, and visualizes bounding boxes."""
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    x = pd.DataFrame(b.transpose(), columns=["x", "y", "width", "height"])

    # seaborn correlogram
    sn.pairplot(x, corner=True, diag_kind="auto", kind="hist", diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / "labels_correlogram.jpg", dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use("svg")  # faster
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    with contextlib.suppress(Exception):  # color histogram bars by class
        [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # known issue #3195
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel("classes")
    sn.histplot(x, x="x", y="y", ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x="width", y="height", ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis("off")

    for a in [0, 1, 2, 3]:
        for s in ["top", "right", "left", "bottom"]:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / "labels.jpg", dpi=200)
    matplotlib.use("Agg")
    plt.close()


def imshow_cls(im, labels=None, pred=None, names=None, nmax=25, verbose=False, f=Path("images.jpg")):
    """Displays a grid of images with optional labels and predictions, saving to a file."""
    from utils.augmentations import denormalize

    names = names or [f"class{i}" for i in range(1000)]
    blocks = torch.chunk(
        denormalize(im.clone()).cpu().float(), len(im), dim=0
    )  # select batch index 0, block by channels
    n = min(len(blocks), nmax)  # number of plots
    m = min(8, round(n**0.5))  # 8 x 8 default
    fig, ax = plt.subplots(math.ceil(n / m), m)  # 8 rows x n/8 cols
    ax = ax.ravel() if m > 1 else [ax]
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i in range(n):
        ax[i].imshow(blocks[i].squeeze().permute((1, 2, 0)).numpy().clip(0.0, 1.0))
        ax[i].axis("off")
        if labels is not None:
            s = names[labels[i]] + (f"—{names[pred[i]]}" if pred is not None else "")
            ax[i].set_title(s, fontsize=8, verticalalignment="top")
    plt.savefig(f, dpi=300, bbox_inches="tight")
    plt.close()
    if verbose:
        LOGGER.info(f"Saving {f}")
        if labels is not None:
            LOGGER.info("True:     " + " ".join(f"{names[i]:3s}" for i in labels[:nmax]))
        if pred is not None:
            LOGGER.info("Predicted:" + " ".join(f"{names[i]:3s}" for i in pred[:nmax]))
    return f


def plot_evolve(evolve_csv="path/to/evolve.csv"):
    """
    Plots hyperparameter evolution results from a given CSV, saving the plot and displaying best results.

    Example: from utils.plots import *; plot_evolve()
    """
    evolve_csv = Path(evolve_csv)
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    x = data.values
    f = fitness(x)
    j = np.argmax(f)  # max fitness index
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc("font", **{"size": 8})
    print(f"Best results from row {j} of {evolve_csv}:")
    for i, k in enumerate(keys[7:]):
        v = x[:, 7 + i]
        mu = v[j]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap="viridis", alpha=0.8, edgecolors="none")
        plt.plot(mu, f.max(), "k+", markersize=15)
        plt.title(f"{k} = {mu:.3g}", fontdict={"size": 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print(f"{k:>15}: {mu:.3g}")
    f = evolve_csv.with_suffix(".png")  # filename
    plt.savefig(f, dpi=200)
    plt.close()
    print(f"Saved {f}")


def plot_results(file="path/to/results.csv", dir=""):
    """
    Plots training results from a 'results.csv' file; accepts file path and directory as arguments.

    Example: from utils.plots import *; plot_results('path/to/results.csv')
    """
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:, j].astype("float")
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # smoothing line
                ax[i].set_title(s[j], fontsize=12)
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.info(f"Warning: Plotting error for {f}: {e}")
    ax[1].legend()
    fig.savefig(save_dir / "results.png", dpi=200)
    plt.close()


def profile_idetection(start=0, stop=0, labels=(), save_dir=""):
    """
    Plots per-image iDetection logs, comparing metrics like storage and performance over time.

    Example: from utils.plots import *; profile_idetection()
    """
    ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)[1].ravel()
    s = ["Images", "Free Storage (GB)", "RAM Usage (GB)", "Battery", "dt_raw (ms)", "dt_smooth (ms)", "real-world FPS"]
    files = list(Path(save_dir).glob("frames*.txt"))
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, ndmin=2).T[:, 90:-30]  # clip first and last rows
            n = results.shape[1]  # number of rows
            x = np.arange(start, min(stop, n) if stop else n)
            results = results[:, x]
            t = results[0] - results[0].min()  # set t0=0s
            results[0] = x
            for i, a in enumerate(ax):
                if i < len(results):
                    label = labels[fi] if len(labels) else f.stem.replace("frames_", "")
                    a.plot(t, results[i], marker=".", label=label, linewidth=1, markersize=5)
                    a.set_title(s[i])
                    a.set_xlabel("time (s)")
                    # if fi == len(files) - 1:
                    #     a.set_ylim(bottom=0)
                    for side in ["top", "right"]:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()
        except Exception as e:
            print(f"Warning: Plotting error for {f}; {e}")
    ax[1].legend()
    plt.savefig(Path(save_dir) / "idetection_profile.png", dpi=200)


def save_one_box(xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True):
    """Crops and saves an image from bounding box `xyxy`, applied with `gain` and `pad`, optionally squares and adjusts
    for BGR.
    """
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix(".jpg"))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
    return crop
