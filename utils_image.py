"""
Functions for encoding and decoding images.
"""
import base64
from io import BytesIO

import cv2
import numpy as np


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
    image_array = cv2.imdecode(array, cv2.IMREAD_ANYCOLOR)  # BGR
    return image_array
