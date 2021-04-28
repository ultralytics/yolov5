import cv2
from pycocotools.coco import COCO

# Create the following folder structure:
# datasets/object365/images/train, datasets/object365/images/val, datasets/labels/train, dataset/labels/val

# Download Object 365 from the Object 365 website And unpack all images in datasets/object365/images/train,
# Put The script and zhiyuan_objv2_train.json file in dataset/object365
# Execute the script in datasets/object365 path

coco = COCO("zhiyuan_objv2_train.json")
cats = coco.loadCats(coco.getCatIds())
nms = [cat["name"] for cat in cats]
print("COCO categories: \n{}\n".format(" ".join(nms)))
cash = set()
for categoryId, cat in enumerate(nms):
    catIds = coco.getCatIds(catNms=[cat])
    imgIds = coco.getImgIds(catIds=catIds)
    print(cat)
    # Create a subfolder in this directory called "labels". This is where the annotations will be saved in YOLO format
    for im in coco.loadImgs(imgIds):
        width, height = im["width"], im["height"]
        path = im["file_name"].split("/")[-1]  # image filename
        try:
            # Test image for missing images
            if path not in cash:
                img = cv2.cvtColor(cv2.imread(f"images/train/{path}"), cv2.COLOR_BGR2RGB)
                cash.add(path)

            with open("labels/train/" + path.replace(".jpg", ".txt"), "a+") as file:
                annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=None)
                for a in coco.loadAnns(annIds):
                    x, y, w, h = a['bbox']  # bounding box in xywh (xy top-left corner)
                    x, y = x + w / 2, y + h / 2  # xy to center
                    file.write(f"{categoryId} {x / width:.5f} {y / height:.5f} {w / width:.5f} {h / height:.5f}\n")

        except Exception as e:
            print(e)
