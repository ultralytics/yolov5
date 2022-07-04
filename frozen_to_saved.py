import tensorflow._api.v2.compat.v1 as tf
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
import time

export_dir = '/home/junaid/yolov5/runs/train/exp_unified_yolov5n_img_cls_more_dense/weights/saved_model/'
graph_pb = '/home/junaid/yolov5/runs/train/exp_unified_yolov5n_img_cls_more_dense/weights/best.pb'

builder = builder.SavedModelBuilder(export_dir)
with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
sigs = {}


name_mappings = {
            "x": "image_arrays",            
            "Identity": "detections",
            "Identity_1": "img_cls_out",
            "Identity_2": "pose_out",
        }

start = time.time()
for i in range(len(graph_def.node)):
    if graph_def.node[i].name in name_mappings.keys():
        graph_def.node[i].name = name_mappings[graph_def.node[i].name]
    print(graph_def.node[i].name)
    for j in name_mappings.keys():
        if j in graph_def.node[i].input:
            for idx, el in enumerate(graph_def.node[i].input):
                if el == j:
                    graph_def.node[i].input[idx] = name_mappings[j]

print('Time taken by replacement: {}'.format(time.time() - start))

with tf.Session(graph=tf.Graph()) as sess:
    # name="" is important to ensure we don't get spurious prefixing
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()
    [print(str(n.name)+str(n.attr['value'].tensor.tensor_shape)) for n in g.as_graph_def().node]


    inp = g.get_tensor_by_name("image_arrays:0")
    out = g.get_tensor_by_name("detections:0")
    out_1 = g.get_tensor_by_name("img_cls_out:0")
    out_2 = g.get_tensor_by_name("pose_out:0")

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = predict_signature_def({"image_arrays": inp}, 
                                                                                        {"detections": out,'img_cls_out':out_1,'pose_out':out_2})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

    
builder.save()

