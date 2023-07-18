from ultralytics import YOLO
import os
from clearml import Dataset, Task
from utils.dataloaders import autosplit
def main():
    # Create a new YOLO model from scratch
    model = YOLO('yolov5n.yaml')

    # Load a pretrained YOLO model (recommended for training)
    # Task.init
    # model = YOLO('yolov5nu.pt')
    # ds = Dataset.get(dataset_id='0442ced245a348008d97adb338f020c6')
    # path_to_local = ds.get_local_copy()
    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    # copy data set to local
    p = '../datasets/fcc_new_no_none'
    if not os.path.exists(p):
        print('downloading dataset...')
        os.makedirs(p)
        ds = Dataset.get(dataset_id='8241c11b8b99472e88c953a30b73eedd')
        ds.get_mutable_local_copy(target_folder=p)
    # list files in the created dir
    os.listdir(p)
    # path_to_local = ds.get_local_copy()
    # p = autosplit(path='/home/tomernahshon/datasets/fcc_new_no_none/images/train', annotated_only=True)

    # task.execute_remotely()
    results = model.train(data='./data/fcc_new_none.yaml', epochs=1,
                          project='fcc-model', name='training_yolov5_full_remote')

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    # results = model('https://ultralytics.com/images/bus.jpg')

    # Export the model to ONNX format
    # success = model.export(format='onnx')


if __name__ == '__main__':
    main()
