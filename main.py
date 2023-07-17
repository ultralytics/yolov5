from ultralytics import YOLO
import os
from clearml import Dataset
def main():
    # Create a new YOLO model from scratch
    # model = YOLO('yolov5n.yaml')

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov5nu.pt')
    ds = Dataset.get(dataset_id='0442ced245a348008d97adb338f020c6')
    path_to_local = ds.get_local_copy()
    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data=os.path.join(path_to_local, 'fcc_new_none.yaml'), epochs=10,
                          project='fcc-model', name='training_yolov5_full_ds1')

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    # results = model('https://ultralytics.com/images/bus.jpg')

    # Export the model to ONNX format
    # success = model.export(format='onnx')


if __name__ == '__main__':
    main()
