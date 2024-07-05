# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Image augmentation functions."""

import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box, xywhn2xyxy
from utils.metrics import bbox_ioa

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, size=640):
        """
        Initializes the Albumentations class for optional data augmentation in YOLOv5 with a specified input size.

        Args:
            size (int, optional): The target size for the augmentation transformations. Defaults to 640.

        Returns:
            None

        Notes:
            This class leverages the `albumentations` library for applying various image augmentation techniques. It requires
            the `albumentations` package (version 1.0.3 or later) to be installed. If the package is not installed or an
            incompatible version is found, the class will silently fail to initialize the transform pipeline.

            Example of setting up Albumentations with custom size:

            ```python
            from ultralytics import Albumentations

            augmentations = Albumentations(size=512)
            ```

            For more information on the `albumentations` library, visit: https://albumentations.ai/docs/
        """
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            T = [
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, im, labels, p=1.0):
        """
        Applies Albumentations transformations to an image and labels with a specified probability.

        Args:
            im (np.ndarray): The input image to be augmented.
            labels (np.ndarray): Array of bounding box labels, where the first column contains class labels and the remaining
                                 columns contain bounding box coordinates in YOLO format.
            p (float): Probability of applying the transformations. Default is 1.0.

        Returns:
            (np.ndarray, np.ndarray): Transformed image and updated labels.
        """
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new["image"], np.array([[c, *b] for c, b in zip(new["class_labels"], new["bboxes"])])
        return im, labels


def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    """
    Applies ImageNet normalization to RGB images in BCHW format, modifying them in-place if specified.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W) representing a batch of images to be normalized.
        mean (tuple[float, float, float]): Mean values for each channel in RGB format. Default is IMAGENET_MEAN (0.485, 0.456, 0.406).
        std (tuple[float, float, float]): Standard deviation values for each channel in RGB format. Default is IMAGENET_STD (0.229, 0.224, 0.225).
        inplace (bool): If True, performs the normalization in-place. Default is False.

    Returns:
        torch.Tensor: Normalized tensor with the same shape and type as input (torch.Tensor).

    Example:
        ```python
        import torch
        from ultralytics import normalize

        x = torch.randn(2, 3, 640, 640)  # Batch of 2 RGB images, each 640x640
        normalized_x = normalize(x)
        ```
    """
    return TF.normalize(x, mean, std, inplace=inplace)


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Reverses the ImageNet normalization on RGB images in BCHW format.

    Args:
        x (torch.Tensor): The BCHW format tensor representing the RGB image to be denormalized.
        mean (tuple[float, float, float], optional): The mean values used for normalization (default is IMAGENET_MEAN).
        std (tuple[float, float, float], optional): The standard deviation values used for normalization (default is IMAGENET_STD).

    Returns:
        torch.Tensor: The denormalized image tensor.

    Example:
        ```python
        import torch
        from your_module import denormalize

        # Assuming x is a BCHW image tensor normalized with ImageNet statistics
        x = torch.randn(1, 3, 224, 224)
        denorm_x = denormalize(x)
        ```
    """
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    """
    Applies HSV color-space augmentation to an image with random gains for hue, saturation, and value.

    Args:
      im (np.ndarray): Input image in BGR format.
      hgain (float): Gain factor for hue adjustment. Default is 0.5.
      sgain (float): Gain factor for saturation adjustment. Default is 0.5.
      vgain (float): Gain factor for value adjustment. Default is 0.5.

    Returns:
      np.ndarray: Augmented image with modified HSV values.

    Notes:
      - The function randomly adjusts the hue, saturation, and value of the input image within specified gain factors.
      - This augmentation helps to improve the robustness of machine learning models by providing varied color-space representations of training images.

    Example:
      ```python
      augmented_image = augment_hsv(image, hgain=0.4, sgain=0.5, vgain=0.6)
      ```
    """
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    """
    Equalizes the histogram of an image, optionally using CLAHE, for BGR or RGB images with range 0-255.

    Args:
      im (np.ndarray): Input image in RGB or BGR format, expected to be a (n, m, 3) array.
      clahe (bool, optional): Flag to use CLAHE (Contrast Limited Adaptive Histogram Equalization) or standard
        histogram equalization. Defaults to True.
      bgr (bool, optional): If True, assumes the input image is in BGR format; otherwise, assumes RGB format.
        Defaults to False.

    Returns:
      np.ndarray: Histogram equalized image of the same shape as input.

    Example:
      ```python
      import cv2
      import numpy as np

      # Load an image in BGR format
      img = cv2.imread('path/to/image.jpg')

      # Apply histogram equalization
      equalized_img = hist_equalize(img, clahe=True, bgr=True)
      ```
    """
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    """
    Replicates half of the smallest object labels in an image for data augmentation.

    Args:
        im (np.ndarray): The input image in which objects are to be replicated.
        labels (np.ndarray): Array of labels, where each label contains the class and bounding box coordinates [class, x1, y1, x2, y2].

    Returns:
        None: The function modifies the input image and labels in-place.

    Notes:
        This function enhances smaller objects by duplicating them randomly in different image regions, which can help improve
        model performance on small objects. The replicated objects are placed at random locations where there is enough space
        for them to fit. Only the smallest 50% of objects are considered for replication.
    ```python
    # Example usage:
    im = cv2.imread('image.jpg')
    labels = np.array([[0, 10, 20, 30, 40], [1, 50, 50, 80, 80]])  # example labels
    replicate(im, labels)
    ```
    """
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[: round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resizes and pads image to the specified new shape with optional stride-multiple constraints.

    Args:
        im (np.ndarray): The input image to be resized and padded.
        new_shape (tuple[int, int] | int): The target shape for the output image. If an integer, a square shape is assumed.
        color (tuple[int, int, int]): The padding color. Defaults to (114, 114, 114).
        auto (bool): If True, makes the padding width and height multiples of the specified stride. Defaults to True.
        scaleFill (bool): If True, stretches the image to fill the new shape. Defaults to False.
        scaleup (bool): If True, allows the image to be upscaled. Defaults to True.
        stride (int): The stride value for padding. Defaults to 32.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The resized and padded image.
            - tuple[float, float]: Width and height scaling ratios.
            - tuple[int, int, int, int]: Padding added to the image (left, top, right, bottom).

    Example:
        ```python
        img = cv2.imread('image.jpg')
        resized_img, ratio, padding = letterbox(img, new_shape=(640, 640))
        ```

    Notes:
        This function is particularly useful for ensuring that input images are correctly resized and padded for
        neural network models like YOLO, which require input images to have consistent dimensions.
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def random_perspective(
    im, targets=(), segments=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)
):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    """
    Applies random perspective transformation to an image, modifying the image and corresponding labels.

    Args:
        im (np.ndarray): Input image to be transformed.
        targets (np.ndarray, optional): Array of target bounding boxes in the format [class, x_min, y_min, x_max, y_max].
        segments (list[np.ndarray], optional): List of segmentation masks for each target object.
        degrees (float, optional): Maximum degrees for image rotation. Default is 10.
        translate (float, optional): Maximum translation as a fraction of image dimensions. Default is 0.1.
        scale (float, optional): Maximum scaling factor. Default is 0.1.
        shear (float, optional): Maximum shearing in degrees. Default is 10.
        perspective (float, optional): Perspective distortion factor. Default is 0.0.
        border (tuple[int, int], optional): Pixel padding added to the border of the image. Default is (0, 0).

    Returns:
        (np.ndarray, np.ndarray): The transformed image and corresponding updated targets.

    Notes:
        - This function uses random parameters for each transformation type (rotation, translation, scaling, shearing,
          perspective) within the specified limits.
        - If segmentation masks are provided, the masks are updated to match the transformed image.
        - Bounding boxes are adjusted to remain within image boundaries after transformation.
        - Transformations are computed in order of translation, shear, rotation, scaling, and perspective to ensure accurate
          results.

    Example:
        ```python
        import cv2
        import numpy as np
        from ultralytics import random_perspective

        # Load image
        image = cv2.imread('image.jpg')
        # Define targets
        targets = np.array([[0, 50, 100, 150, 200]])
        # Apply random perspective transformation
        transformed_image, transformed_targets = random_perspective(image, targets)
        ```

        Note that `transformed_image` contains the augmented image and `transformed_targets` contains the updated bounding
        box coordinates.

    References:
        - `torchvision.transforms.RandomAffine`: https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomAffine
        - YOLOv5: https://github.com/ultralytics/ultralytics
    """
    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments) and len(segments) == n
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    """
    Applies Copy-Paste augmentation by flipping and merging segments and labels on an image.

    Details at https://arxiv.org/abs/2012.07177.

    Args:
        im (np.ndarray): The input image array with shape (height, width, channels).
        labels (np.ndarray): An array of labels corresponding to objects in the image, in the format (class, x1, y1, x2, y2).
        segments (list[np.ndarray]): A list of segmentation masks, each as an array of shape (num_points, 2).
        p (float): Probability of applying the Copy-Paste augmentation, with a default value of 0.5.

    Returns:
        None: The function modifies the input image and labels in-place.

    Notes:
        - This function modifies the input image and labels directly without returning them.
        - It uses flipping to augment segments and merges them into the provided labels and image.

    Example:
        ```python
        augmented_image, new_labels = copy_paste(image, labels, segments, p=0.6)
        ```
    """
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED)

        result = cv2.flip(im, 1)  # augment segments (flip left-right)
        i = cv2.flip(im_new, 1).astype(bool)
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def cutout(im, labels, p=0.5):
    """
    Applies cutout augmentation to an image with optional label adjustment, using random masks of varying sizes.

    Args:
      im (np.ndarray): The input image in which cutout augmentation will be applied.
      labels (np.ndarray): Array of bounding box labels with shape (n, 5), where each label contains [class, x, y, width, height].
      p (float): Probability of applying cutout augmentation. Default is 0.5.

    Returns:
      np.ndarray: The augmented image with cutout applied.
      np.ndarray: The potentially adjusted bounding box labels.
    """
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, xywhn2xyxy(labels[:, 1:5], w, h))  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    """
    Applies MixUp augmentation by blending images and labels.

    See https://arxiv.org/pdf/1710.09412.pdf for details.

    Args:
      im (np.ndarray): First input image as a NumPy array.
      labels (np.ndarray): Labels associated with the first image. Typically, it has shape (N, 5) where N is the number of labels,
                           and each label consists of [class, x_center, y_center, width, height].
      im2 (np.ndarray): Second input image as a NumPy array.
      labels2 (np.ndarray): Labels associated with the second image, following the same structure as `labels`.

    Returns:
      (tuple): Tuple containing:
        - im (np.ndarray): MixUp augmented image.
        - labels (np.ndarray): Combined labels from both input images.

    Example:
      ```python
      augmented_im, augmented_labels = mixup(image1, labels1, image2, labels2)
      ```
    """
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    """
    Filters bounding box candidates by minimum width-height thresholds, aspect ratio, and area ratio, ensuring valid
    bounding boxes post-augmentation.

    Args:
        box1 (np.ndarray): Array of shape (4, n) representing bounding boxes before augmentation.
        box2 (np.ndarray): Array of shape (4, n) representing bounding boxes after augmentation.
        wh_thr (float, optional): Minimum width and height threshold in pixels. Defaults to 2.
        ar_thr (float, optional): Maximum aspect ratio threshold. Defaults to 100.
        area_thr (float, optional): Minimum area ratio threshold. Defaults to 0.1.
        eps (float, optional): Small epsilon value to avoid division by zero. Defaults to 1e-16.

    Returns:
        np.ndarray: A boolean array indicating valid bounding box candidates post-augmentation.

    Examples:
        ```python
        box1 = np.array([[10, 20, 30, 40], [15, 25, 35, 45]]).T
        box2 = np.array([[12, 22, 32, 42], [18, 28, 38, 48]]).T
        candidates = box_candidates(box1, box2, wh_thr=2, ar_thr=50, area_thr=0.2)
        ```

    Notes:
        - This function is typically used in image augmentation pipelines to filter out invalid bounding boxes that may
          arise due to various geometric transformations.
        - Ensure that the input bounding boxes are in the format [x_min, y_min, x_max, y_max].
        - Aspect ratio is calculated to ensure bounding boxes are not disproportionately skewed.
        - Width and height thresholds ensure that only sufficiently large bounding boxes are considered valid.
        - The area ratio threshold ensures the area of the augmented box remains significant relative to the original.

    References:
        - Bounding box filtering techniques: https://arxiv.org/pdf/1506.02640.pdf
        - Image augmentation with bounding boxes: https://arxiv.org/pdf/1708.04896.pdf
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def classify_albumentations(
    augment=True,
    size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.0 / 0.75),  # 0.75, 1.33
    hflip=0.5,
    vflip=0.0,
    jitter=0.4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    auto_aug=False,
):
    # YOLOv5 classification Albumentations (optional, only used if package is installed)
    """
    Sets up and returns Albumentations transforms for YOLOv5 classification tasks depending on augmentation settings.

    Args:
        augment (bool): Whether to apply data augmentation. Default is True.
        size (int): Target size for the input images. Default is 224.
        scale (tuple[float, float]): Scaling factor range for RandomResizedCrop. Default is (0.08, 1.0).
        ratio (tuple[float, float]): Aspect ratio range for RandomResizedCrop. Default is (0.75, 1.0 / 0.75).
        hflip (float): Probability of applying horizontal flip. Default is 0.5.
        vflip (float): Probability of applying vertical flip. Default is 0.0.
        jitter (float): Color jitter intensity. Default is 0.4.
        mean (tuple[float, float, float]): Mean values for normalization. Default is IMAGENET_MEAN.
        std (tuple[float, float, float]): Standard deviation values for normalization. Default is IMAGENET_STD.
        auto_aug (bool): Whether to apply auto augmentation policies like AugMix, AutoAug, or RandAug. Default is False.

    Returns:
        albumentations.Compose: A composed Albumentations transform pipeline for classification tasks.

    Notes:
        - Auto augmentations such as AugMix, AutoAug, and RandAug are currently not supported.
        - Requires `albumentations` package installed. Install it with `pip install albumentations`.

    Example:
        ```python
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        from ultralytics import classify_albumentations

        transforms = classify_albumentations(augment=True, size=224, hflip=0.5)
        augmented = transforms(image=image)
        ```
    """
    prefix = colorstr("albumentations: ")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        check_version(A.__version__, "1.0.3", hard=True)  # version requirement
        if augment:  # Resize and crop
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio)]
            if auto_aug:
                # TODO: implement AugMix, AutoAug & RandAug in albumentation
                LOGGER.info(f"{prefix}auto augmentations are currently not supported")
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if jitter > 0:
                    color_jitter = (float(jitter),) * 3  # repeat value for brightness, contrast, saturation, 0 hue
                    T += [A.ColorJitter(*color_jitter, 0)]
        else:  # Use fixed crop for eval set (reproducibility)
            T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size, width=size)]
        T += [A.Normalize(mean=mean, std=std), ToTensorV2()]  # Normalize and convert to Tensor
        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        return A.Compose(T)

    except ImportError:  # package not installed, skip
        LOGGER.warning(f"{prefix}⚠️ not found, install with `pip install albumentations` (recommended)")
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")


def classify_transforms(size=224):
    """
    Applies a series of transformations including center crop, ToTensor, and normalization for image classification.

    Args:
        size (int): The size to which the image will be resized. This must be an integer.

    Returns:
        torchvision.transforms.Compose: A composed transform object that applies the following:
            - Converts images to PyTorch tensors
            - Resizes images to the specified size
            - Centers and crops images to the specified size
            - Normalizes images using ImageNet mean and standard deviation

    Examples:
        ```python
        import torchvision.transforms as T
        from ultralytics import classify_transforms

        transform = classify_transforms(size=224)
        ```
    """
    assert isinstance(size, int), f"ERROR: classify_transforms size {size} must be integer, not (list, tuple)"
    # T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class LetterBox:
    # YOLOv5 LetterBox class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, size=(640, 640), auto=False, stride=32):
        """
        Initializes a LetterBox object for YOLOv5 image preprocessing with optional auto sizing and stride adjustment.

        Args:
          size (tuple[int, int] | int): Desired output size. If an integer is provided, it will be used for both height and width.
          auto (bool): When set to True, automatically adjusts the size while maintaining stride constraints (Default is False).
          stride (int): Adjusts padding to be a multiple of the stride value (Default is 32).

        Returns:
          None

        Notes:
          This class is typically used as part of a preprocessing pipeline for images to be fed into a YOLOv5 model.
          It facilitates resizing and padding images to maintain aspect ratio and adhere to model input size constraints.
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):
        """
        Resizes and pads input image `im` (HWC format) to specified dimensions, maintaining aspect ratio.

        Args:
            im (np.ndarray): Input image with shape (height, width, channels).

        Returns:
            np.ndarray: Resized and padded image.

        Steps:
        1. Calculate the resizing ratio based on the target height and width relative to the input image dimensions.
        2. Determine the new dimensions after resizing to maintain the aspect ratio.
        3. Optionally adjust dimensions based on the configured stride to ensure that the dimensions are multiples of `stride`.
        4. Calculate the padding needed to center the resized image within the target dimensions.
        5. Create an output image filled with a constant value (114) to act as the background padding.
        6. Place the resized image in the center of the output image, thus applying the padding on the sides where necessary.

        Examples:
            ```python
            import cv2
            import numpy as np

            # Initialize the LetterBox object
            lb = LetterBox(size=(640, 640), auto=True, stride=32)

            # Read an example image
            image = cv2.imread("example.jpg")

            # Apply the LetterBox transformation
            result_image = lb(image)
            ```

        Notes:
            - The above example demonstrates reading an image from a file, resizing, and padding it for input into YOLOv5.
            - The parameter `auto` helps automatically adjust the padding according to the given stride. This is particularly
              useful for ensuring that the resulting image dimensions are compatible with the network requirements.
        """
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old
        h, w = round(imh * r), round(imw * r)  # resized image
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else self.h, self.w
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


class CenterCrop:
    # YOLOv5 CenterCrop class for image preprocessing, i.e. T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, size=640):
        """
        Initializes the CenterCrop object for image preprocessing, accepting a single integer or a tuple for the target
        size, with a default value of 640.

        Args:
            size (int | tuple): Target size for center cropping. If a single integer is provided, it will be used for both
                height and width. If a tuple is provided, it should be in the form (height, width). Defaults to 640.

        Returns:
            None

        Example:
            ```python
            from torchvision.transforms import Compose
            from ultralytics import CenterCrop

            # Example of CenterCrop within a Compose transformation
            transforms = Compose([CenterCrop(size=320), ...])
            image = transforms(image)
            ```
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):
        """
        Provides a center crop transformation to the input image, ensuring the output size is as specified.

        Args:
            im (np.array): Input image in HWC format to be center-cropped.

        Returns:
            np.array: Center-cropped image; cropped from the center to match the height and width specified during
                      initialization.

        Notes:
            - The aspect ratio of the input image is maintained after cropping.
            - The cropping dimensions are determined based on the smaller dimension of the input image to ensure a
              centered result.

        Example:
            ```python
            center_crop = CenterCrop(size=(640, 480))
            cropped_image = center_crop(input_image)
            ```
        """
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ToTensor:
    # YOLOv5 ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, half=False):
        """
        Initializes the ToTensor object for YOLOv5 image preprocessing, converting numpy arrays to PyTorch tensors.

        Args:
            half (bool): If True, converts the numpy array to a half-precision (FP16) PyTorch tensor. Defaults to False.

        Returns:
            None: This method does not return any value.

        Examples:
            ```python
            to_tensor = ToTensor(half=True)
            tensor_image = to_tensor(image)
            ```
        """
        super().__init__()
        self.half = half

    def __call__(self, im):
        """
        Converts an image from HWC format to CHW format, and normalizes it to the [0, 1] range.

        Args:
          im (np.ndarray): Input image in HWC format with channels in BGR order.

        Returns:
          torch.Tensor: Normalized image tensor in CHW format with channels in RGB order. The dtype is either float32 or float16
              based on the initialization parameter 'half'.

        Notes:
          This transformation is essential for preparing an image for model inference, aligning it with the expected input
          format of PyTorch models. If 'half' is set to True during initialization, the function outputs a tensor with
          half-precision (FP16), allowing for faster computation on compatible hardware like NVIDIA's Tensor Cores.
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
