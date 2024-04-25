import torch
import cv2
from yolov5 import YOLOv5

def main():
    # Load the pre-trained YOLOv5 model
    model_path = "yolov5s.pt"  # Assuming you have the 'yolov5s' model
    model = YOLOv5(model_path)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Starting the video stream. Press 'q' to exit.")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform inference
        results = model.predict(frame)

        # Draw results on the frame
        annotated_frame = results.render()[0]

        # Display the resulting frame
        cv2.imshow('YOLOv5 Real-Time Obstacle Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
