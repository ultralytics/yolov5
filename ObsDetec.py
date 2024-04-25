import cv2
from deepsparse import compile_model
from deepsparse.utils import yolov5_utils

# Load your ONNX model
onnx_filepath = "/home/talha/yolov5/yolov5s.onnx"
model = compile_model(model_path=onnx_filepath, batch_size=1)

# Set up webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img, ratio, (dw, dh) = yolov5_utils.preprocess(frame, model.inputs[0].shape[2], model.inputs[0].shape[3])
    img = img.transpose((2, 0, 1))[None,]

    # Run inference
    outputs = model.run([img])[0]

    # Postprocess the outputs
    detections = yolov5_utils.postprocess(outputs, ratio, (dw, dh))

    # Draw the detections on the frame
    frame = yolov5_utils.draw_detections(frame, detections)

    # Display the resulting frame
    cv2.imshow('Real-Time Obstacle Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the capture
cap.release()
cv2.destroyAllWindows()
