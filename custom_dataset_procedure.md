## custom data set training with YOLOV5 using LOCO as example

### prepare dataset
#### loco dataset
original loco is COCO format
location: /home/epi/gary/mydata/loco/
#### data visualization
data/load_loco.ipynb

#### convert COCO to YOLOV5 format
- annoation path is hard coded in python script
    /home/epi/gary/github/JSON2YOLO/general_json2yolo.py
- why 5097 found, 496 missing ??

#### what YOLOV5 need
- dataset config file 
    data/loco.yaml
     
    '''
    for both train & val image data
    /home/epi/gary/mydata/loco/images/
    '''
    and a label path looks like */labels/ to contain all YOLOV5 txt labels
    '''
    labels in text
    /home/epi/gary/mydata/loco/labels/
    ```

- model config file
    models/yolov5s_loco.yaml
    this is almost identical to yolov5s.yaml with only number of class changed: nc=5

### start to train
```
# loco 5 classes 
# train.sh
python train.py --img 640 --batch 8 --epochs 100 --data ./data/loco.yaml --cfg ./models/yolov5s_loco.yaml --weights './yolov5s.pt' --device 0

```