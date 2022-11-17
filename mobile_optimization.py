import os

import cv2
import numpy as np
import tensorflow as tf

saved_model_dir = "/home/parvej/Downloads/seami_models/best_saved_model/"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

filename = 'seami_10c_860epoch.tflite'
with open(filename, "wb") as f:
    f.write(tflite_quant_model)

# dirpath = "/home/parvej/Videos/images/"
# files = os.listdir(dirpath)
# print(files)
# BATCH_SIZE = 1
# NORM_H = NORM_W = 320
# def rep_data_gen():
#     a = []
#     for filename in files:
#         img = cv2.imread(dirpath + filename)
#         img = cv2.resize(img, (NORM_H, NORM_W))
#         img = img / 255.0
#         img = img.astype(np.float32)
#         a.append(img)
#     a = np.asarray(a).astype(np.float32)
#     print(a.shape) # a is np array of 160 3D images
#     img = tf.data.Dataset.from_tensor_slices(a).batch(1)
#     for i in img.take(BATCH_SIZE):
#         print(i)
#         yield [i]
# # https://www.tensorflow.org/lite/performance/post_training_quantization
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8
# converter.representative_dataset=rep_data_gen
# tflite_quant_model = converter.convert()
# tflite_model_file = "new_model1.tflite"
# with open(tflite_model_file, "wb") as f:
#     f.write(tflite_quant_model)
# tflite_quant_model.write_bytes(tflite_quant_model)