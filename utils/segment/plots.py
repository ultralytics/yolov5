import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from ..general import xywh2xyxy
from ..plots import Annotator, colors


def plot_masks(img, masks, colors, alpha=0.5):
    """
    Args:
        img (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
        masks (tensor): predicted masks on cuda, shape: [n, h, w]
        colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
    Return:
        ndarray: img after draw masks, shape: [h, w, 3]

    transform colors and send img_gpu to cpu for the most time.
    """
    img_gpu = img.clone()
    num_masks = len(masks)
    if num_masks == 0:
        return img.permute(1, 2, 0).contiguous().cpu().numpy() * 255

    # [n, 1, 1, 3]
    # faster this way to transform colors
    colors = torch.tensor(colors, device=img.device).float() / 255.0
    colors = colors[:, None, None, :]
    # [n, h, w, 1]
    masks = masks[:, :, :, None]
    masks_color = masks.repeat(1, 1, 1, 3) * colors * alpha
    inv_alph_masks = masks * (-alpha) + 1
    masks_color_summand = masks_color[0]
    if num_masks > 1:
        inv_alph_cumul = inv_alph_masks[:(num_masks - 1)].cumprod(dim=0)
        masks_color_cumul = masks_color[1:] * inv_alph_cumul
        masks_color_summand += masks_color_cumul.sum(dim=0)

    # print(inv_alph_masks.prod(dim=0).shape) # [h, w, 1]
    img_gpu = img_gpu.flip(dims=[0])  # filp channel for opencv
    img_gpu = img_gpu.permute(1, 2, 0).contiguous()
    # [h, w, 3]
    img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
    return (img_gpu * 255).byte().cpu().numpy()


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    import random

    # Plots one bounding box on image img
    tl = (line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def plot_images_and_masks(
    images,
    targets,
    masks,
    paths=None,
    fname="images.jpg",
    names=None,
    max_size=640,
    max_subplots=16,
):
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
        masks = masks.astype(int)

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            idx = (targets[:, 0]).astype(int)
            image_targets = targets[idx == i]

            if masks.max() > 1.0:  # mean that masks are overlap
                image_masks = masks[[i]]  # (1, 640, 640)
                # convert masks (1, 640, 640) -> (n, 640, 640)
                nl = len(image_targets)
                index = np.arange(nl).reshape(nl, 1, 1) + 1
                image_masks = np.repeat(image_masks, nl, axis=0)
                image_masks = np.where(image_masks == index, 1.0, 0.0)
            else:
                image_masks = masks[idx == i]

            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype("int")
            labels = image_targets.shape[1] == 6  # labels if no conf column
            conf = (None if labels else image_targets[:, 6])  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale_factor < 1:  # absolute coords need scale if image scales
                    boxes *= scale_factor
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = colors(cls)
                cls = names[cls] if names else cls
                if scale_factor < 1:
                    mask = image_masks[j].astype(np.uint8)
                    mask = cv2.resize(mask, (w, h))
                    mask = mask.astype(np.bool)
                else:
                    mask = image_masks[j].astype(np.bool)
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = "%s" % cls if labels else f"{cls} {conf[j]:.1f}"
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)
                    mosaic[block_y:block_y + h, block_x:block_x +
                           w, :][mask] = mosaic[block_y:block_y + h, block_x:block_x + w, :][mask] * 0.35 + (
                               np.array(color) * 0.65)

        # Draw image filename labels
        if paths:
            label = Path(paths[i]).name[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(
                mosaic,
                label,
                (block_x + 5, block_y + t_size[1] + 5),
                0,
                tl / 3,
                [220, 220, 220],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

        # Image border
        cv2.rectangle(
            mosaic,
            (block_x, block_y),
            (block_x + w, block_y + h),
            (255, 255, 255),
            thickness=3,
        )

    if fname:
        r = min(1280.0 / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save
        with Image.fromarray(mosaic) as im:
            im.save(fname)
    return mosaic


def plot_results_with_masks(file="path/to/results.csv", dir="", best=True):
    # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for _, f in enumerate(files):
        try:
            data = pd.read_csv(f)
            index = np.argmax(
                0.9 * data.values[:, 8] + 0.1 * data.values[:, 7] + 0.9 * data.values[:, 12] +
                0.1 * data.values[:, 11],)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 7, 8, 11, 12]):
                y = data.values[:, j]
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=2)
                if best:
                    # best
                    ax[i].scatter(
                        index,
                        y[index],
                        color="r",
                        label=f"best:{index}",
                        marker="*",
                        linewidth=3,
                    )
                    ax[i].set_title(s[j] + f"\n{round(y[index], 5)}")
                else:
                    # last
                    ax[i].scatter(x[-1], y[-1], color="r", label="last", marker="*", linewidth=3)
                    ax[i].set_title(s[j] + f"\n{round(y[-1], 5)}")
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print(f"Warning: Plotting error for {f}: {e}")
    ax[1].legend()
    fig.savefig(save_dir / "results.png", dpi=200)
    plt.close()
