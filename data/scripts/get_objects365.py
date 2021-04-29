# Objects365 https://www.objects365.org labels JSON to YOLO script
# 1. Download Object 365 from the Object 365 website And unpack all images in datasets/object365/images
# 2. Place this file and zhiyuan_objv2_train.json file in datasets/objects365
# 3. Execute this file from datasets/object365 path
# /datasets
#     /objects365
#         /images
#         /labels

from pycocotools.coco import COCO

coco = COCO("zhiyuan_objv2_train.json")
cats = coco.loadCats(coco.getCatIds())
nms = [cat["name"] for cat in cats]
print("COCO categories: \n{}\n".format(" ".join(nms)))
for categoryId, cat in enumerate(nms):
    catIds = coco.getCatIds(catNms=[cat])
    imgIds = coco.getImgIds(catIds=catIds)
    print(cat)
    # Create a subfolder in this directory called "labels". This is where the annotations will be saved in YOLO format
    for im in coco.loadImgs(imgIds):
        width, height = im["width"], im["height"]
        path = im["file_name"].split("/")[-1]  # image filename
        try:
            with open("labels/train/" + path.replace(".jpg", ".txt"), "a+") as file:
                annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=None)
                for a in coco.loadAnns(annIds):
                    x, y, w, h = a['bbox']  # bounding box in xywh (xy top-left corner)
                    x, y = x + w / 2, y + h / 2  # xy to center
                    file.write(f"{categoryId} {x / width:.5f} {y / height:.5f} {w / width:.5f} {h / height:.5f}\n")

        except Exception as e:
            print(e)
