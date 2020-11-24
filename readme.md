# Adapt YOLOv5 to detect bike lane traffic signs

##TASKS and MODIFICATIONS

### Transfer lerning
- Create alternative training scenarios so we can finetune the model with freezed layers as per [Ultralytics:#1314](https://github.com/ultralytics/yolov5/issues/1314)
  - train_ft_fc.py --> Freeze all except the last fully connected layer
  - train_ft_backbone.py --> Freeze the backbone and train the rest of the network
  
### Data augmentation
Yolov% uses Mosaic by default, it's not suitable for detection of traffic signs as the composition of the images create unrealistic options.
- Deactivate Mosaic
- Take into account the transformations that make sense for each class
- Using library [imgaug](https://imgaug.readthedocs.io/en/latest/)


  
  
