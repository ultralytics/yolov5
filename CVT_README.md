


## Tagged validation

This is just a draft for now: 



```python
python yolov5/val_tagged.py 
--data data/validation-tagged.yaml
--weights weights/last-purple_boot_3l6p24vb.pt
--batch-size 1
--task val
--save-txt-and-json
```
The output structure is the same as when running yolo validation, but now we have an extra 
`labels_tagged` folder.

`labels_tagged` is a folder containing JSON files, one for each input image, where the name of each JSON file 
corresponds to the ID of the corresponding input image, e.g. `TMX7316010203-000045_pano_0000_004370.json`


The JSON file contains information about the ground truth (GT) and true positive (TP) bounding boxes and their 
associated labels.

`GT_boxes` - contains a list of lists, where each inner list represents a bounding box in the ground truth with four 
values: 

`[normalized x-coordinate, normalized y-coordinate, normalized width, normalized height]`

`GT_labels` - contains a list of integers, where each integer represents the tagged label of the corresponding 
bounding box in the `GT_boxes` field.

`TP_labels` - contains a list of integers, where each integer represents the label of the corresponding bounding box 
that is considered a true positive.

In each JSON we provide information about the bounding boxes detected in each input image.
This allows us to evaluate the performance of an object detection model based on the ground truth and true positive 
bounding boxes.