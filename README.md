# YOLOv5 for Computer Vision Team - Amsterdam

This repository is a fork of the YOLOv5 implementation customized for the use of the Computer Vision Team in Amsterdam. It builds upon the YOLOv5 framework and introduces custom modifications to work with Azure Machine Learning, database integrations and tagged validation.

For the original YOLOv5 documentation, please refer to [YOLOv5 README](YOLOv5_README.md).

## Installation

#### 1. Clone the code

```bash
git clone git@github.com:Computer-Vision-Team-Amsterdam/yolov5.git
```

#### 2. Install Poetry
If you don't have it yet, follow the instructions [here](https://python-poetry.org/docs/#installation) to install the package manager Poetry.

#### 3. Install dependencies
In the terminal, navigate to the project root (the folder containing `pyproject.toml`), then use Poetry to create a new virtual environment and install the dependencies.

```bash
poetry install
```
### Modifications 

#### Tagged validation

To analyze bias in our data, we divided the categories "person" and "license_plate" into smaller groups. This is called "tagged validation". We added tags for gender, age, and skin tone to each label. Each combination of these tags was given a unique number, like 1 for "man/child/dark," 2 for "man/child/medium," and so on.

Then, we included these numbers in our YOLO annotations. This way, we can see how well our model works in different situations by looking at the validation results.

The input structure is the same as when running yolo validation, but now the 
labels files contain an extra column with the tagged class id. 

Example of one txt file in `data/labels/val`:
```
0 0.6237570155750621 0.5660134381788937 0.02723971280184667 0.11326378807027071 2
0 0.6031440235076742 0.5605816356638536 0.020922329336564238 0.11149882760277408 5
```

The indices from the 6th column are coming from the Azure COCO annotation file
export from Data Labelling tool.

To run validation:

```python
python yolov5/val_tagged.py 
--data PATH_TO_YAML_FILE
--weights PATH_TO_WEIGHTS_FILE.pt
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

#### Database

To access a database in Azure Machine Learning it is necessary to create a `database.json` file inside the `database` folder.
An example of the structure can be found in the folder under the name `database.example.json`.

This database.json file should include the following information:
``` 
    client_id:      client id of the managed identity in Azure
```

