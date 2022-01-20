# Training Steps
## Activate the virtual environments
The following command assume that the virtual python environment are installed in `~/virtual_env`.

```bash
source ~/virtual_env/bin/activate
```
## Install the libraries

```bash
git clone --recursive https://github.com/fringe-ai/LMI_AI_Solutions.git
git clone -b YJ https://github.com/fringe-ai/yolov5.git
cd yolov5
pip install -r requirements.txt
```
## Activate LMI_AI environment
The following commands assume that the LMI_AI_Solution repo is cloned in `~/LMI_AI_Solutions`.

```bash
source ~/LMI_AI_Solutions/lmi_ai.env
```

## Prepare the datasets
Prepare the datasets by the followings:
- resize images to 640 x 640
- convert to YOLO annotation format

Assume that the original annotated files are in `./data/allImages_1024`. After execting the exmaple commands below, it will generate a yolo formatted folder in `./data/2022-01-08_640_yolo`.

Below are the example commands:
```bash
python -m resize_images_with_csv -i ./data/allImages_1024 --out_imsz 640,640 -o ./data/2022-01-08_640
python ./preprocess/convert_data_to_yolo.py -i ./data/2022-01-08_640 -o ./data/2022-01-08_640_yolo
```

## Create a yaml file indicating the locations of datasets
Note that the order of class names in the yaml file must match with the order of names in this json file: `./data/2022-01-08_640_yolo/class_map.json`.
```yaml
path: ./data/2022-01-04_640_yolo  # dataset root dir
train: images  # train images (relative to 'path')
val: images  # val images (relative to 'path')
test:  # test images (optional)
 
# Classes
nc: 3  # number of classes
names: ['peeling', 'scuff', 'white']  # class names must match with the names in class_map.json
```
Let's save the yaml file as `./config/2022-01-08_640.yaml`

## Download the pre-trained yolo model on COCO dataset
The pre-trained yolo models can be found in: https://github.com/ultralytics/yolov5/releases/tag/v6.0. 
The following commands show that 
- download [the pre-trained model (yolov5s.pt)](https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt)
- move it to `./pretrained-models`.

```bash
mkdir ./pretrained-models
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -P ./pretrained-models
```

## Train the model with custom datasets
The command below trains the datasets in the yaml file with the following arguments:
- img: image size
- batch: batch size
- data: the path to the yaml file
- weights: the path to the pre-trained weights file
- project: the output folder
- name: the subfolder to be created inside the output folder
- exist-ok: if it's okay to overwrite the existing output subfolder

```bash
python train.py --img 640 --batch 16 --epoch 600 --data ./config/2022-01-08_640.yaml --weights ./pretrained-models/yolov5s.pt --project ./training --name 2022-01-08_640 --exist-ok
```

## Monitor the training progress
```bash
tensorboard --logdir ./training/2022-01-08_640
```
Execuate the command above and go to http://localhost:6006 to monitor the training.



# Testing
After training, the weights are saved in `./training/2022-01-08_640/weights/best.pt`. Copy the best.pt to `./trained-inference-models/2022-01-08_640`.

```bash
cp ./training/2022-01-08_640/weights/best.pt ./trained-inference-models/2022-01-08_640
```

## Run inference
The command below run the inference using the following arguments:
- source: the path to the test images
- weights: the path to the trained model weights file
- img: the image size
- project: the output folder
- name: the subfolder to be created inside the output folder
- save-csv: save the outputs as a csv file

```bash
python detect.py --source ./data/2022-01-04_640_yolo/images --weights ./trained-inference-models/2022-01-05_640/best.pt --img 640 --project ./validation --name 2022-01-08_640 --save-csv
```
The output results are saved in `./validation/2022-01-08_640`.
