# Objects365 https://www.objects365.org labels JSON to YOLO script
# 1. Download Object 365 from the Object 365 website And unpack all images in datasets/object365/images
# 2. Place this file and zhiyuan_objv2_train.json file in datasets/objects365
# 3. Execute this file from datasets/object365 path
# /datasets
#     /objects365
#         /images
#         /labels


from pycocotools.coco import COCO

from utils.general import download, Path

# Make Directories
dir = Path('../datasets/objects365')  # dataset directory
for p in 'images', 'labels':
    (dir / p).mkdir(parents=True, exist_ok=True)
    for q in 'train', 'val':
        (dir / p / q).mkdir(parents=True, exist_ok=True)

# Download
url = "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/"
download([url + 'zhiyuan_objv2_train.tar.gz'], dir=dir)  # annotations json
download([url + f for f in [f'patch{i}.tar.gz' for i in range(51)]], dir=dir / 'images' / 'train', curl=True, threads=8)

# Labels
coco = COCO(dir / 'zhiyuan_objv2_train.json')
names = [x["name"] for x in coco.loadCats(coco.getCatIds())]
for categoryId, cat in enumerate(names):
    catIds = coco.getCatIds(catNms=[cat])
    imgIds = coco.getImgIds(catIds=catIds)
    for im in coco.loadImgs(imgIds):
        width, height = im["width"], im["height"]
        path = Path(im["file_name"])  # image filename
        try:
            with open(dir / 'labels' / 'train' / path.with_suffix('.txt').name, 'a') as file:
                annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=None)
                for a in coco.loadAnns(annIds):
                    x, y, w, h = a['bbox']  # bounding box in xywh (xy top-left corner)
                    x, y = x + w / 2, y + h / 2  # xy to center
                    file.write(f"{categoryId} {x / width:.5f} {y / height:.5f} {w / width:.5f} {h / height:.5f}\n")

        except Exception as e:
            print(e)
