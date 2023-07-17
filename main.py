from ultralytics import YOLO
def main():
    # Create a new YOLO model from scratch
    # model = YOLO('yolov5n.yaml')

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov5nu.pt')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='/home/tomernahshon/yolov5/data/fcc_new_none.yaml', epochs=2000,
                          project='fcc-model', name='training_yolov5_full_ds')

    # Evaluate the model's performance on the validation set
    # results = model.val()

    # Perform object detection on an image using the model
    # results = model('https://ultralytics.com/images/bus.jpg')

    # Export the model to ONNX format
    # success = model.export(format='onnx')


if __name__ == '__main__':
    main()
