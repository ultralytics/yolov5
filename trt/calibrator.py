import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import glob

import ctypes
import logging
logger = logging.getLogger(__name__)
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


"""
There are 4 types calibrator in TensorRT.
trt.IInt8LegacyCalibrator
trt.IInt8EntropyCalibrator
trt.IInt8EntropyCalibrator2
trt.IInt8MinMaxCalibrator
"""

class Calibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, stream, cache_file=""):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.stream = stream
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = cache_file
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        print("######################")
        print(names)
        print("######################")
        batch = self.stream.next_batch()
        if not batch.size:   
            return None

        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)

def preprocess(img, input_shape, letter_box=False):
    """Preprocess an image before TRT YOLO inferencing.
    """
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 114, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img

class DataLoader:
    def __init__(self, batch_size, batch_num, calib_img_dir, input_w, input_h):
        self.index = 0
        self.length = batch_num
        self.batch_size = batch_size
        self.input_h = input_h
        self.input_w = input_w
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(calib_img_dir, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, \
            '{} must contains more than '.format(calib_img_dir) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size, 3, input_h, input_w), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess(img, (self.input_h, self.input_w, 3), letter_box=True)
                self.calibration_data[i] = img

            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length