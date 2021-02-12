import numpy as np
import onnxruntime

x = np.zeros((1, 3, 320, 320)).astype('float32')

session = onnxruntime.InferenceSession("yolov5s.onnx", None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(input_name, output_name)

result = session.run([output_name], {input_name: x})
print(result[0].shape)