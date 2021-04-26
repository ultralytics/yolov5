from pycocotools.coco import COCO
import cv2
import numpy as np
import glob
import shutil

# Truncates numbers to N decimals
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


# Create the following folder structure dataset/object365/images/train, dataset/object365/images/val, dataset/labels/images/train, dataset/labels/images/val

# Download Object 365 from the Object 365 website And unpack all images in dataset/object365/images/train,Put The script and zhiyuan_objv2_train.json file in dataset/object365
# Execute the script in dataset/object365v path


coco = COCO("zhiyuan_objv2_train.json")
cats = coco.loadCats(coco.getCatIds())
nms = [cat["name"] for cat in cats]
print("COCO categories: \n{}\n".format(" ".join(nms)))
cash = set()

for categoryId, cat in enumerate(nms):
    catIds = coco.getCatIds(catNms=[cat])
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)
    print(cat)
    # # Create a subfolder in this directory called "labels". This is where the annotations will be saved in YOLO format
    for im in images:
        dw = 1.0 / im["width"]
        dh = 1.0 / im["height"]

        annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        path = im["file_name"].split("/")
        fixed_path = path[-1]
        filename = fixed_path.replace(".jpg", ".txt")

        try:
            # Test image for missing images
            if fixed_path not in cash:
                img = cv2.imread(f"images/train/{fixed_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cash.add(fixed_path)

            with open("labels/train/" + filename, "a+") as myfile:
                for i in range(len(anns)):
                    xmin = anns[i]["bbox"][0]
                    ymin = anns[i]["bbox"][1]
                    xmax = anns[i]["bbox"][2] + anns[i]["bbox"][0]
                    ymax = anns[i]["bbox"][3] + anns[i]["bbox"][1]

                    x = (xmin + xmax) / 2
                    y = (ymin + ymax) / 2

                    w = xmax - xmin
                    h = ymax - ymin

                    x = x * dw
                    w = w * dw
                    y = y * dh
                    h = h * dh

                    # Note: This assumes a single-category dataset, and thus the "0" at the beginning of each line.
                    mystring = f"{categoryId} {truncate(x, 7)} {truncate(y, 7)} {truncate(w, 7)} {truncate(h, 7)}"
                    myfile.write(mystring)
                    myfile.write("\n")
            myfile.close()

        except Exception as e:
            print(e)

current_dir = "images/train"
chances = 10  # 10% val set
for fullpath in glob.iglob(os.path.join(current_dir, "*.jpg")):
    n = random.randint(1, 100)
    if n <= chances:
        title, ext = os.path.splitext(os.path.basename(fullpath))
        print(title)
        shutil.move(f"images/train/{title}.jpg", f"images/val/{title}.jpg")
        shutil.move(f"labels/train/{title}.txt", f"labels/val/{title}.txt")
