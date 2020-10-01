"""
Functions for encoding and decoding images.
"""
import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def encode_image(image):
    """Encode an image to base64 encoded bytes.
    Args:
        image: PIL.PngImagePlugin.PngImageFile
    Returns:
        base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format="png")
    base64_bytes = base64.b64encode(buffered.getvalue())
    return base64_bytes.decode("utf-8")


def decode_image(field):
    """Decode a base64 encoded image to a list of floats.
    Args:
        field: base64 encoded string
    Returns:
        numpy.array
    """
    array = np.frombuffer(base64.b64decode(field), dtype=np.uint8)
    image_array = cv2.imdecode(array, cv2.IMREAD_ANYCOLOR)
    # output_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return image_array


def superimpose_heatmap(img, mask):
    """Superimpose mask heatmap on image."""
    heatmap = cv2.applyColorMap(np.uint8(mask * 255), cv2.COLORMAP_JET)
    new_img = np.float32(heatmap) / 255 + np.float32(img)
    new_img = new_img / np.max(new_img)
    new_img = np.uint8(255 * new_img)
    return Image.fromarray(new_img)


def get_heatmap(norm_attr):
    """Get heatmap from array with values between 0 and 1."""
    heatmap = cv2.applyColorMap(np.uint8(norm_attr * 255), cv2.COLORMAP_JET)
    return Image.fromarray(heatmap)


def _normalize_scale(attr, scale_factor):
    if abs(scale_factor) < 1e-5:
        return np.clip(attr, -1, 1)
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)


def _cumulative_sum_threshold(values, percentile):
    # given values should be non-negative
    assert 0 <= percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]


def normalize_image_attr(attr, sign="absolute_value", outlier_perc=2):
    attr_combined = np.sum(attr, axis=2)
    if sign == "all":
        threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
    elif sign == "positive":
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif sign == "negative":
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif sign == "absolute_value":
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    return _normalize_scale(attr_combined, threshold)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image."""
    buf = BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def deprocess_image(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-8)
    img = img * 0.05
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.clip(img * 255, 0, 255).astype('uint8')
