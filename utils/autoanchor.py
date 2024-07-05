# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""AutoAnchor utils."""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils import TryExcept
from utils.general import LOGGER, TQDM_BAR_FORMAT, colorstr

PREFIX = colorstr("AutoAnchor: ")


def check_anchor_order(m):
    """
    Checks and corrects the anchor order against stride in YOLOv5 `Detect` module if necessary.

    Args:
        m (Detect): The YOLOv5 `Detect` module containing anchor and stride information.

    Returns:
        None: This function does not return any value.

    Notes:
        The function computes the mean anchor area for each output layer and verifies their order against the strides.
        If the order is inconsistent, it reverses the anchor order to ensure proper matching with the strides.

    Example:
        ```python
        from models.yolo import Model
        from utils.autoanchor import check_anchor_order

        model = Model(cfg)  # initialize your YOLOv5 model here
        check_anchor_order(model.model[-1])  # check and correct anchor order for the Detect module
        ```
    """
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        LOGGER.info(f"{PREFIX}Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)


@TryExcept(f"{PREFIX}ERROR")
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    """
    Evaluates anchor fit to a given dataset and adjusts the model's anchors if necessary, supporting customizable
    threshold and image size.

    Args:
        dataset (Dataset): Dataset containing images and labels to evaluate against the model's anchors.
        model (torch.nn.Module): YOLOv5 model whose anchors are to be evaluated and updated.
        thr (float, optional): Threshold for anchor evaluation. Default is 4.0.
        imgsz (int, optional): Image size for evaluation. Default is 640.

    Returns:
        None

    Notes:
        The function assesses the current anchors by computing metrics like the ratio metric, anchors above threshold,
        and best possible recall (BPR). If the BPR exceeds 0.98, the anchors are considered a good fit. Otherwise, it
        attempts to improve the anchors using k-means clustering. If new anchors are better, they replace the old ones in
        the model. The new anchors should be manually updated in the model configuration file (*.yaml) for future use.

    Example:
        ```python
        from ultralytics import YOLO

        # Load dataset and model
        dataset = YOLO('coco128.yaml')
        model = YOLO('yolov5s.pt')

        # Check and adjust anchors
        check_anchors(dataset, model)
        ```

    See Also:
        - kmean_anchors: Function for generating anchors using k-means clustering.
        - check_anchor_order: Function for checking and correcting the anchor order against the model's stride.
    """
    m = model.module.model[-1] if hasattr(model, "module") else model.model[-1]  # Detect()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        """Computes ratio metric, anchors above threshold, and best possible recall for YOLOv5 anchor evaluation."""
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1 / thr).float().mean()  # best possible recall
        return bpr, aat

    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # model strides
    anchors = m.anchors.clone() * stride  # current anchors
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f"\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). "
    if bpr > 0.98:  # threshold to recompute
        LOGGER.info(f"{s}Current anchors are a good fit to dataset ✅")
    else:
        LOGGER.info(f"{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...")
        na = m.anchors.numel() // 2  # number of anchors
        anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= stride
            s = f"{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)"
        else:
            s = f"{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)"
        LOGGER.info(s)


def kmean_anchors(dataset="./data/coco128.yaml", n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """
    Creates k-means-evolved anchors from a training dataset, optionally with genetic algorithm evolution.

    Args:
        dataset (str | dict): Path to the dataset YAML file or a loaded dataset. Default is './data/coco128.yaml'.
        n (int): Number of anchors to generate. Default is 9.
        img_size (int): Image size to use for k-means clustering. Default is 640.
        thr (float): Anchor-label width-height ratio threshold hyperparameter. Default is 4.0.
        gen (int): Number of generations for genetic algorithm evolution of anchors. Default is 1000.
        verbose (bool): If True, prints detailed results and progress. Default is True.

    Returns:
        np.ndarray: A numpy array of shape (n, 2) containing the evolved anchors.

    Notes:
        This function first attempts to generate initial anchors using k-means clustering on the dataset labels, and then
        optionally improves these anchors through multiple generations of a genetic algorithm.

    Examples:
        ```python
        from utils.autoanchor import kmean_anchors
        anchors = kmean_anchors(dataset='./data/coco.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True)
        ```
    """
    from scipy.cluster.vq import kmeans

    npr = np.random
    thr = 1 / thr

    def metric(k, wh):  # compute metrics
        """Computes ratio metric, anchors above threshold, and best possible recall for YOLOv5 anchor evaluation."""
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        """Evaluates fitness of YOLOv5 anchors by computing recall and ratio metrics for an anchor evolution process."""
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k, verbose=True):
        """Sorts and logs kmeans-evolved anchor metrics and best possible recall values for YOLOv5 anchor evaluation."""
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = (
            f"{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n"
            f"{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, "
            f"past_thr={x[x > thr].mean():.3f}-mean: "
        )
        for x in k:
            s += "%i,%i, " % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k

    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors="ignore") as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.dataloaders import LoadImagesAndLabels

        dataset = LoadImagesAndLabels(data_dict["train"], augment=True, rect=True)

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f"{PREFIX}WARNING ⚠️ Extremely small objects found: {i} of {len(wh0)} labels are <3 pixels in size")
    wh = wh0[(wh0 >= 2.0).any(1)].astype(np.float32)  # filter > 2 pixels
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans init
    try:
        LOGGER.info(f"{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...")
        assert n <= len(wh)  # apply overdetermined constraint
        s = wh.std(0)  # sigmas for whitening
        k = kmeans(wh / s, n, iter=30)[0] * s  # points
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        LOGGER.warning(f"{PREFIX}WARNING ⚠️ switching strategies from kmeans to random init")
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), bar_format=TQDM_BAR_FORMAT)  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f"{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}"
            if verbose:
                print_results(k, verbose)

    return print_results(k).astype(np.float32)
