Forked Yolov5 repository to be used by Computer Vision Team.
See the README of Yolov5 [here](YOLOv5_README.md).

## Development

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

The input structure is the same as when running yolo validation, but now the 
labels files contain an extra column with the tagged class id. 

Example of one txt file in `data/labels/val`:
```
0 0.6237570155750621 0.5660134381788937 0.02723971280184667 0.11326378807027071 2
0 0.6031440235076742 0.5605816356638536 0.020922329336564238 0.11149882760277408 5
```

The indices from the 6th column are coming from the Azure COCO annotation file
export from Data Labelling tool. First, we 

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

## Database

To access the database is necessary to create a `database.json` file inside the `database` folder.
An example of the structure can be found in the folder under the name `database.example.json`.

The file contains the following information:
``` 
    hostname:       hostname address of the database
    username:       managed identity name in production
    database_name
    client_id:      client id of the managed identity
```
