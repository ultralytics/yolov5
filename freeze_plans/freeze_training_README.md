# This tutorial is for freeze training

## What for
There may be a need for some peoples to train a model with some structure freezed.

## How to use
Customize the model training process by defining a yaml file.

### Demo
```
freeze_train.py --weights yolov5s.pt \
--data data/coco128.yaml --cfg models/yolov5s.yaml --batch-size -1 \
--device 0,1,2,3 --hyp data/hyps/hyp.finetune.yaml --cache \
--freeze-plan freeze_plans/freeze_exp.yaml
```

### TODO
- [ ] Support selecting part of the data in any freeze training step to train the model.

## Implementation
Currently the training process is independent from train.py, because it needs a little bit more time to put it into train.py, and I am not sure if this feature is important for this repository for now. If necessary, I can merge the changes into train.py later, and I will maintain the bugs of this feature in time.