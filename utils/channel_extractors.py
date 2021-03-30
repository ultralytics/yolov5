import cv2


def extract_red(frame):
    return _extract_one_idx(frame, 2)

def extract_green(frame):
    return _extract_one_idx(frame, 1)

def extract_blue(frame):
    return _extract_one_idx(frame, 0)

def extract_gray(frame):
    pass

def extract_hue(frame):
    pass

def _extract_one_idx(frame, idx):
    return frame[..., [idx]]