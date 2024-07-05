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
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        """
        Initializes the Colors class with a palette derived from the Ultralytics color scheme, converting hex codes to
        RGB.

        Colors derived from `hex = matplotlib.colors.TABLEAU_COLORS.values()`.

        Returns
        -------
        None

        Notes
        -----
        This class sets up a predefined RGB color palette to be used throughout the library, enabling consistent color
        usage in visualizations. The palette is generated from a series of hex color codes and converted to RGB format
        for compatibility with various plotting libraries. This standardization helps maintain a cohesive visual style
        across different outputs.
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
        """
        Returns a color from the predefined Ultralytics color palette by index.

        Args:
            i (int): Index of the color in the color palette.
            bgr (bool): If True, the color is returned in BGR format. If False, the color is returned in RGB format.

        Returns:
            (tuple[int, int, int]): A tuple representing the color in the specified format (RGB or BGR).

        Examples:
            ```python
            colors = Colors()
            color_rgb = colors(0)  # Returns the first color in RGB format
            color_bgr = colors(0, bgr=True)  # Returns the first color in BGR format
            ```
        """
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """
        Converts a hexadecimal color code to an RGB tuple.

        Args:
            h (str): A string representing a hexadecimal color code. Example: '#FF5733'.

        Returns:
            tuple: A tuple (int, int, int) representing the RGB color values.

        Examples:
            ```python
            rgb_color = Colors.hex2rgb('#FF5733')
            print(rgb_color)  # Output: (255, 87, 51)
            ```
        """
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    """
    Provides visualization and saving of feature maps for model layers during training or inference.

    Args:
        x (torch.Tensor): Tensor containing the feature maps to be visualized, with shape (batch, channels, height, width).
        module_type (str): Type of the module (e.g., 'Detect', 'Segment') which determines the visualization context.
        stage (int): The stage or layer number of the model from which the feature maps are extracted.
        n (int, optional): The maximum number of feature maps to visualize. Defaults to 32.
        save_dir (pathlib.Path, optional): The directory where the visualization images will be saved. Defaults to Path("runs/detect/exp").

    Returns:
        None

    Notes:
        - Only modules not of type 'Detect' or 'Segment' will have their features visualized in this function.
        - The function saves visualization plots as PNG files in the specified directory.
        - Visualization is constrained to the first example in the batch (`x[0]`).

    Examples:
        ```python
        from pathlib import Path
        import torch

        # Assuming 'x' is the feature map tensor from a model
        x = torch.rand((1, 64, 128, 128))  # Example shape: (batch_size, num_channels, height, width)
        feature_visualization(x, 'Backbone.Conv', 1, n=16, save_dir=Path('runs/detect/exp1'))
        ```
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
    Generates a logarithmic 2D histogram.

    This function is useful for visualizing distributions, such as label or evolution distributions, by creating a 2D histogram
    with logarithmic scaling of bin counts.

    Args:
        x (np.ndarray): The x-coordinates of the data points.
        y (np.ndarray): The y-coordinates of the data points.
        n (int, optional): The number of bins to use for the histogram in both dimensions. Defaults to 100.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the histogram array, x-bin edges, and y-bin edges.

    Example:
        ```python
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        hist, xedges, yedges = hist2d(x, y, n=100)
        ```
    """
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    """
    Applies a zero-phase low-pass Butterworth filter to a signal.

    Args:
      data (np.ndarray): Input signal to be filtered.
      cutoff (float, optional): Cutoff frequency of the filter in Hz. Defaults to 1500.
      fs (float, optional): Sampling frequency of the input signal in Hz. Defaults to 50000.
      order (int, optional): Order of the Butterworth filter. Higher order means a steeper roll-off. Defaults to 5.

    Returns:
      np.ndarray: Filtered signal after applying the low-pass Butterworth filter.

    Notes:
      Employs a forward-backward filter using `scipy.signal.filtfilt` to eliminate phase distortion in the filter output.

    Example:
      ```python
      import numpy as np
      from your_module import butter_lowpass_filtfilt

      # Create a sample signal with noise
      fs = 50000  # Sampling frequency
      t = np.linspace(0, 1, fs, endpoint=False)  # Time vector
      freq = 1234  # Frequency of the signal
      x = np.sin(2 * np.pi * freq * t) + 0.5 * np.random.randn(t.size)

      # Apply the low-pass Butterworth filter
      cutoff = 1500
      filtered_signal = butter_lowpass_filtfilt(x, cutoff, fs)
      ```
    ```
    """
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
    """
    Converts YOLOv5 model output to a target format suitable for plotting and analysis.

    Args:
        output (torch.Tensor): The YOLOv5 model output tensor. Each element is expected to have object detection data
                               including bounding box coordinates, confidence scores, and class IDs.
        max_det (int): The maximum number of detections to consider for each image. Defaults to 300.

    Returns:
        torch.Tensor: A tensor containing the converted targets in the format [batch_id, class_id, x_center, y_center,
                      width, height, confidence]. Each row represents a detected object.

    Notes:
        This function splits the output tensor from the YOLOv5 model into bounding boxes, confidence scores, and class IDs.
        It then converts the bounding box format from [x_min, y_min, x_max, y_max] to [x_center, y_center, width, height]
        before combining these with the batch ID and class ID into a single tensor.

    Example:
        ```python
        output = [
            torch.tensor([[50, 30, 200, 180, 0.8, 1], [70, 40, 210, 190, 0.7, 0]]),  # Example data
            torch.tensor([[60, 35, 205, 185, 0.6, 1]])
        ]
        targets = output_to_target(output)
        ```
    """
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, xyxy2xywh(box), conf), 1))
    return torch.cat(targets, 0).numpy()


@threaded
def plot_images(images, targets, paths=None, fname="images.jpg", names=None):
    """
    Plot an image grid with labels from YOLOv5 predictions or targets and save the output image.

    Args:
        images (numpy.ndarray | torch.Tensor): Batch of images to plot, expected shape (B, C, H, W).
        targets (numpy.ndarray | torch.Tensor): Corresponding targets for the images. Shape and content depend on model outputs.
        paths (list[str] | None): Optional list of file paths for each image. Default is None.
        fname (str): Filename to save the plotted image. Default is "images.jpg".
        names (list[str] | None): Optional list of class names corresponding to target classes. Default is None.

    Returns:
        None

    Notes:
        - The function supports both PyTorch Tensors and NumPy arrays for `images` and `targets`.
        - Images are scaled and annotated before being saved as a mosaic grid in the specified `fname`.

    Examples:
        ```python
        import torch
        from ultralytics import plot_images

        images = torch.rand((4, 3, 640, 640))  # Batch of 4 random images
        targets = torch.tensor([])  # No targets for simplicity
        plot_images(images, targets, fname="output.jpg")
        ```
    """
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
    for i in range(i + 1):
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
    """
    Plots the learning rate schedule for the provided optimizer and scheduler, saving the plot to the specified
    directory.

    Args:
    optimizer (torch.optim.Optimizer): The optimizer for which the learning rate schedule will be plotted.
    scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler that will modify the learning rate of the optimizer.
    epochs (int): The number of epochs to simulate (default is 300).
    save_dir (str | Path): The directory to save the resulting plot (default is an empty string, which refers to the current directory).

    Returns:
    None

    Notes:
    - This function copies the optimizer and scheduler to prevent modifications to the original objects.

    Examples:
    ```python
    # Define optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Plot learning rate schedule
    plot_lr_scheduler(optimizer, scheduler, epochs=100, save_dir="results/")
    ```

    ```python
    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Plot learning rate schedule
    plot_lr_scheduler(optimizer, scheduler, epochs=150, save_dir="output/")
    ```
    """
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

    Args:
        None

    Returns:
        None

    Example:
        ```python
        from utils.plots import plot_val_txt
        plot_val_txt()
        ```

    Note:
        This function reads bounding box data from 'val.txt' and generates histograms using matplotlib. Ensure 'val.txt' is
        present in the working directory.
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
    Plot histograms of object detection targets from 'targets.txt', saving the figure as 'targets.jpg'.

    Args:
        None

    Returns:
        None

    Example:
        ```python
        from utils.plots import plot_targets_txt
        plot_targets_txt()
        ```

    Notes:
        The 'targets.txt' file should contain columns of detection target data in the format [x, y, width, height].

    The function reads the 'targets.txt' file, processes the detection targets, and plots histograms to visualize the distributions of x, y coordinates, widths, and heights. The resulting figure is saved as 'targets.jpg'.
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
    Plot validation study results from 'study*.txt' files to compare model performance and speed.

    Args:
        file (str): Specific file path for plotting (default: "").
        dir (str): Directory containing 'study*.txt' files (default: "").
        x (np.ndarray | None): Optional x-axis values for custom plotting (default: None).

    Returns:
        None

    Example:
        ```python
        from utils.plots import plot_val_study
        plot_val_study(file='path/to/study_result.txt', dir='path/to/results')
        ```

    Notes:
        - This function loads validation results from specified or default text files and visualizes the comparison
          of model performance metrics such as Precision, Recall, mAP, and Inference Times.
        - If `plot2` is set to True, it will generate additional subplot visualizations for detailed metric comparisons.
        - The function saves the output plot image to the `save_dir` with the filename 'study.png'.
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
    """
    Plots dataset labels, saving correlogram and label images, handles classes, and visualizes bounding boxes.

    Args:
        labels (np.ndarray): Array of labels with the first column as class indices and remaining columns as bounding box coordinates.
        names (tuple): Optional tuple of class names.
        save_dir (Path): Directory to save the output plots. Defaults to Path("").

    Returns:
        None

    Notes:
        - This function generates a seaborn correlogram of the labels, showing relationships among bounding box properties.
        - A histogram of instances per class is generated, with bars colored according to the class palette.
        - Bounding boxes are scaled and drawn on a white canvas to visualize their distribution.
        - Outputs are saved as 'labels_correlogram.jpg' and 'labels.jpg' in the specified `save_dir`.

    Examples:
        ```python
        import numpy as np
        from pathlib import Path
        from utils.plots import plot_labels

        labels = np.random.rand(100, 5)  # Example labels
        plot_labels(labels, save_dir=Path("./output"))
        ```
    """
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
    """
    Displays a grid of images with optional labels and predictions, saving the visualization to a file.

    Args:
      im (torch.Tensor): Batch of images to display, typically shape (N, C, H, W).
      labels (torch.Tensor | None): True class labels for each image, shape (N,). Defaults to None.
      pred (torch.Tensor | None): Predicted class labels for each image, shape (N,). Defaults to None.
      names (list[str] | None): List of class names for indexing the labels and predictions. Defaults to None.
      nmax (int): Maximum number of images to display in the grid. Defaults to 25.
      verbose (bool): If True, logs additional information about the saved images and labels. Defaults to False.
      f (Path | str): File path to save the resulting image grid. Defaults to Path("images.jpg").

    Returns:
      None

    Notes:
      The function utilizes the denormalize transformation to convert tensor values back to a visualizable range.
      Class labels and predictions, if provided, are displayed as the titles of the subplots.

    Examples:
      ```python
      import torch
      from pathlib import Path

      # Example inputs
      images = torch.randn(16, 3, 224, 224)  # Randomly generated images
      true_labels = torch.randint(0, 10, (16,))  # Random true labels
      predicted_labels = torch.randint(0, 10, (16,))  # Random predicted labels
      class_names = [f"class{i}" for i in range(10)]  # List of class names

      # Saving visualization
      imshow_cls(images, labels=true_labels, pred=predicted_labels, names=class_names, f=Path("output.jpg"))
      ```

      Logging information, including true and predicted class labels, is displayed if `verbose` is set to True.
    """
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

    Args:
        evolve_csv (str): Path to the CSV file containing the hyperparameter evolution results.

    Returns:
        None

    Example:
        ```python
        from utils.plots import plot_evolve
        plot_evolve('path/to/evolve.csv')
        ```

    Notes:
        For more information on hyperparameter evolution in Ultralytics, refer to:
        https://docs.ultralytics.com/yolov5/tutorials/hyperparameter_evolution
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
    Plot training results from a 'results.csv' file, visualizing metrics for better model performance understanding.

    Args:
        file (str): Path to the results CSV file. Defaults to 'path/to/results.csv'.
        dir (str): Directory containing results CSV files. Defaults to an empty string.

    Returns:
        None

    Example:
        ```python
        from utils.plots import plot_results
        plot_results('path/to/results.csv')
        ```
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
    Plots per-image iDetection logs, comparing various metrics over time.

    Args:
        start (int, optional): Starting index of the data to plot. Defaults to 0.
        stop (int, optional): Stopping index of the data to plot. If 0, plots till the end. Defaults to 0.
        labels (tuple, optional): Labels for the plots, one per file. If empty, uses filenames as labels. Defaults to ().
        save_dir (str | Path, optional): Directory where the log files are stored. Defaults to "".

    Returns:
        None

    Example:
        ```python
        from ultralytics.utils.plotting import profile_idetection
        profile_idetection(start=0, stop=1000, labels=('Label1', 'Label2'), save_dir='logs')
        ```
    ``
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
    """
    Crops and saves an image from a bounding box 'xyxy', applying optional transformations.

    Args:
        xyxy (Tensor | ArrayLike): Bounding box in (x1, y1, x2, y2) format.
        im (ndarray): Input image array.
        file (Path | str, optional): Path to save the cropped image. Defaults to `Path("im.jpg")`.
        gain (float, optional): Factor by which to scale the bounding box. Defaults to 1.02.
        pad (int, optional): Padding to add around the bounding box. Defaults to 10.
        square (bool, optional): If True, converts the bounding box to a square. Defaults to False.
        BGR (bool, optional): If True, handles image in BGR format; otherwise, uses RGB. Defaults to False.
        save (bool, optional): If True, saves the cropped image to disk. Defaults to True.

    Returns:
        ndarray: Cropped image array (if `save` is False).

    Example:
        ```python
        import cv2
        from pathlib import Path
        from ultralytics.utils.plots import save_one_box

        image = cv2.imread("input_image.jpg")
        bbox = [50, 50, 200, 200]
        save_one_box(bbox, image, file=Path("cropped_image.jpg"))
        ```

    Notes:
    - Automatically creates the necessary directories if they do not exist.
    - Adjusts bounding box coordinates and clips values to the image dimensions to avoid out-of-bound errors.
    - When saving, addresses potential chroma subsampling issues by using PIL to ensure quality preservation.
    - For more information, see: [Ultralytics Documentation](https://github.com/ultralytics/ultralytics).
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
