"""
yolo로 추출한 그래프를 feature vector화 하는 코드
일회성임.

"""

import numpy as np

import pickle
from sklearn.preprocessing import normalize
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow.keras.layers as layers

with open("../graph_filenames.bin", "rb") as f:
    filename_list = pickle.load(f)

# 정제한 그래프들로만 feature vector 뽑아냄

from tqdm import notebook
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

batch_size = 100
GRAPH_DIR = "runs/detect/exp10/"

fvecs = []

img_list = []
batch_size = 100
input_shape = (224, 224, 3)

nloop = math.ceil(len(filename_list) / batch_size)

base = tf.keras.applications.MobileNetV2(
    input_shape=input_shape, include_top=False, weights="imagenet"
)
base.trainable = False

model = Model(inputs=base.input, outputs=layers.GlobalAveragePooling2D()(base.output))


def preprocess_img(img):
    X = np.asarray(img).astype(np.float32)
    X = tf.image.resize(X, input_shape[:2])
    X = preprocess_input(X)
    return X


# img_list.append(X)

for k in tqdm(range(nloop)):

    img_list = []

    for f in filename_list[
        k * batch_size : min((k + 1) * batch_size, len(filename_list))
    ]:
        img = plt.imread(GRAPH_DIR + f)
        img_list.append(preprocess_img(img))

    list_ds = tf.data.Dataset.from_tensor_slices(img_list)

    dataset = list_ds.batch(batch_size).prefetch(-1)

    for batch in dataset:
        fvecs_batch = model.predict(batch)
    fvecs.extend(fvecs_batch)

with open("representative_graph_fvecs.bin", "wb") as f:
    pickle.dump(fvecs, f)
