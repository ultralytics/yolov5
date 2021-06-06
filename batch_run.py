import cv2
import boto3
import io
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox
from utils.general import non_max_suppression
from dotenv import load_dotenv, find_dotenv
from pathlib import Path  # Python 3.6+ only
import os
import mysql.connector
from nanoid import generate

env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

prod = os.getenv("PROD")
mydb = None
BUCKET_NAME = None
if(prod == "True"):
    database_name = "filtariprod"
    mydb = mysql.connector.connect(
    host="filtari-prod-database.cxxmrtwgw1xk.us-east-1.rds.amazonaws.com",
    user="admin",
    password= os.getenv("PROD_DB"),
    database= database_name)
    BUCKET_NAME = 'filtari-post-images-prod'
else:
    database_name = "innodb"
    mydb = mysql.connector.connect(
    host="filtari-qa-database-2.cxxmrtwgw1xk.us-east-1.rds.amazonaws.com",
    user="admin",
    password= os.getenv("QA_DB"),
    database= database_name)
    BUCKET_NAME = 'filtari-post-images-qa'
cursor = mydb.cursor()


s3 = boto3.resource('s3', region_name='us-east-1')
bucket = s3.Bucket(BUCKET_NAME)
ALCOHOL_FLAG = 2
device = select_device('')
model = attempt_load('./drinkNetV2.pt', map_location=device)  # load FP32 model
half = device.type != 'cpu'  # half precision only supported on CUDA
stride = int(model.stride.max())  # model stride
#options for ML
class opt: pass
opt.augment = None
opt.weights = "./drinkNetV2.pt"
opt.conf_thres = 0.25
opt.iou_thres = 0.45 
opt.classes = None
opt.agnostic_nms = None

class CustomLoadImages:  # for inference
    def __init__(self, keys, img_size=640, stride=32):
        ni, nv = len(keys), 0

        self.img_size = img_size
        self.stride = stride
        self.keys = keys
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        key = self.keys[self.count]
        # Read image
        self.count += 1
        img0 = getImageFromS3(key)
        try:
            assert img0 is not None, 'Image Not Found ' + key
        except:
            return key, None, None, None
        print(f'image {self.count}/{self.nf} {key}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return key, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

# Let's use Amazon S3
def getImageFromS3(key):
    print("getting image from s3 with key", key)
    img_object = bucket.Object(key)
    file_stream = io.BytesIO(img_object.get()['Body'].read())
    return cv2.imdecode(np.frombuffer(file_stream.read(), np.uint8), 1)

def detect(dataset):
    results = []
    for key, img, im0s, vid_cap in dataset:
        if img is None:
            print("skipping key {} because it couldn't be read in".format(key))
        else:
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    results.append(1)
                else:
                    results.append(0)
    return results

def update_scans(scan_ids, status):
    query = "UPDATE `{}`.`scan` SET`status` = '{}' WHERE `scan_id` in (".format(database_name,status)
    for idx,val in enumerate(scan_ids):
        query += f'{val}'
        if(idx < len(scan_ids)-1):
            query += ','
    query+= ');'
    cursor.execute(query)
    mydb.commit()

def bulk_insert(results):
    query = "INSERT INTO `{}`.`post_after_ml_mapping` (`post_id`,`category_id`,`type`) values ".format(database_name)
    for idx,val in enumerate(results):
        query += f'({val[1]},{val[0]},\'image_ml\')'
        if(idx < len(results)-1):
            query += ','
    query+= ';'
    cursor.execute(query)
    mydb.commit()

if __name__ == '__main__':
    cursor.execute("SELECT scan_id from scan where scan.status = '{}';".format('image_ml'))
    scan_ids = []
    for i in cursor:
      scan_ids.append(i[0])
    print("scan ids", scan_ids)
    if(not len(scan_ids) == 0):
        batch_name = "process_batch_" + generate()
        print("batch name", batch_name)
        update_scans(scan_ids, batch_name)
        cursor.execute("SELECT scan.scan_id,image_s3_link,post_id FROM post join scan on post.scan_id = scan.scan_id where post.is_primary = 1 and post.image_s3_link != 'none' and scan.status = '{}'".format(batch_name))
        keys = []
        post_ids = []
        scan_id_mapping = {}
        current_scan_id = None
        for i in cursor:
            ## logic is for having multiple scans
            if(current_scan_id == None):
                current_scan_id = i[0]
            elif(not current_scan_id == i[0]):
                scan_id_mapping[current_scan_id] = (keys, post_ids)
                current_scan_id = i[0]
                keys = []
            splitLink = i[1].split("/")
            print("link", splitLink)
            key = "{}/{}/{}".format(splitLink[3], splitLink[4], splitLink[5])
            keys.append(key)
            post_ids.append(i[2])
        ## adding the last set of keys
        if(not current_scan_id == None):
            scan_id_mapping[current_scan_id] = (keys, post_ids)
        print("keys", keys)
        for scan_key in scan_id_mapping:
            keys = scan_id_mapping[scan_key][0]
            post_ids = scan_id_mapping[scan_key][1]
            dataset = CustomLoadImages(keys)
            results = detect(dataset)
            flags_to_insert = []
            for idx,val in enumerate(results):
                if(val == 1):
                    flags_to_insert.append((ALCOHOL_FLAG, post_ids[idx]))
            print("scan id:{} flagged post ids:{} ".format(scan_key, flags_to_insert))
            bulk_insert(flags_to_insert)
        update_scans(scan_ids, "pre_review")
        print("job complete")
