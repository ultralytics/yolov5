# Training Steps
## Activate the virtual environments
The following command assume that the virtual python environment are installed in `~/virtual_env`.

```bash
source ~/virtual_env/bin/activate
```
## Install the libraries

```bash
git clone --recursive https://github.com/lmitechnologies/LMI_AI_Solutions.git
cd object_detectors/yolov5
pip install -r requirements.txt
```
## Activate LMI_AI environment
The following commands assume that the LMI_AI_Solution repo is cloned in `~/LMI_AI_Solutions`, and assume that it's the working directory, unless otherwise mentioned.

```bash
cd ~/LMI_AI_Solutions
source lmi_ai.env
source object_detectors/yolov5/yolo.env
```

## Prepare the datasets
The example dataset can be downloaded using GCP:
```
engagements/nordson/chattanooga/catheter/feasibility/models/pytorch/defeat/objdet/yolov5/data/allImages_1024
```

Prepare the datasets by the followings:
- resize images to 640 x 640
- convert to YOLO annotation format

Assume that the original annotated files are downloaded in `./data/allImages_1024`. After execting the exmaple commands below, it will generate a yolo formatted folder in `./data/resized_640_yolo`.

```bash
python -m resize_with_csv --path_imgs ./data/allImages_1024 --out_imsz 640,640 --path_out ./data/resized_640
python -m convert_data_to_yolo --path_imgs ./data/resized_640 --path_out ./data/resized_640_yolo
```

## Create a yaml file indicating the locations of datasets
Note that the order of class names in the yaml file must match with the order of names in this json file: `./data/resized_640_yolo/class_map.json`.
```yaml
path: /home/yijun.jiang/projects/LMI_AI_Solutions/data/resized_640_yolo  # dataset root dir (must use absolute path!)
train: images  # train images (relative to 'path')
val: images  # val images (relative to 'path')
test:  # test images (optional)
 
# Classes
nc: 3  # number of classes
names: ['peeling', 'scuff', 'white']  # class names must match with the names in class_map.json
```
Let's save the yaml file as `./config/example.yaml`

## Download the pre-trained yolo model
The pre-trained yolo models can be found in: https://github.com/ultralytics/yolov5/releases/tag/v6.0. 
The following command shows that 
- download [the pre-trained model (yolov5s.pt)](https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt)
- save it to `./pretrained-models`.

```bash
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -P ./pretrained-models
```

## Train the model
The command below trains the datasets in the yaml file with the following arguments:
- img: image size
- batch: batch size
- data: the path to the yaml file
- rect(optinal): if the images are rectangular
- weights: the path to the pre-trained weights file
- project: the output folder
- name: the subfolder to be created inside the output folder
- exist-ok(optional): if it's okay to overwrite the existing output subfolder

```bash
python -m train --img 640 --batch 16 --epoch 600 --data ./config/example.yaml --weights ./pretrained-models/yolov5s.pt --project ./training --name example --exist-ok
```

## Monitor the training progress
```bash
tensorboard --logdir ./training/example
```
While training process is running, open another terminal.
Execuate the command above and go to http://localhost:6006 to monitor the training.


# Testing
## Save trained model
After training, the weights are saved in `./training/example/weights/best.pt`. Copy the best.pt to `./trained-inference-models/example`.

```bash
mkdir -p ./trained-inference-models/example
cp ./training/example/weights/best.pt ./trained-inference-models/example
```

## Run inference
The command below run the inference using the following arguments:
- source: the path to the test images
- weights: the path to the trained model weights file
- img: the image size
- project: the output folder
- conf-thres(optional): the confidence level, default is 0.25
- name: the subfolder to be created inside the output folder
- save-csv: save the outputs as a csv file

```bash
python -m detect --source ./data/resized_640_yolo/images --weights ./trained-inference-models/example/best.pt --img 640 --project ./validation --name example --save-csv
```
The output results are saved in `./validation/example`.
