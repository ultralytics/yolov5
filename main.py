import os
import tkinter as tk

import cv2
import easyocr
import pytesseract
from PIL import Image, ImageTk

from detect import main, parse_opt

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"


object_class = [
    {
        "person": 0,
        "bicycle": 1,
        "car": 2,
        "motorcycle": 3,
        "airplane,": 4,
        "bus": 5,
        "train": 6,
        "truck": 7,
        "boat": 8,
        "traffic light": 9,
        "fire hydrant": 10,
        "stop sign": 11,
        "parking meter": 12,
        " bench": 13,
        "bird": 14,
        "cat": 15,
        "dog": 16,
        "horse": 17,
        "sheep": 18,
        "cow": 19,
        "elephant": 20,
        "bear": 21,
        "zebra": 22,
        "giraffe": 23,
        "backpack": 24,
        "umbrella": 25,
        "handbag": 26,
        "tie": 27,
        "suitcase": 28,
        "frisbee": 29,
        "skis": 30,
        "snowboard": 31,
        "sports ball": 32,
        "kite": 33,
        "baseball bat": 34,
        "baseball glove": 35,
        "skateboard": 36,
        "surfboard": 37,
        "tennis racket": 38,
        "bottle": 39,
        "wine glass": 40,
        "cup": 41,
        "fork": 42,
        "knife": 43,
        "spoon": 44,
        "bowl ": 45,
        "banana": 46,
        "apple": 47,
        "sandwich": 48,
        "orange": 49,
        "broccoli": 50,
        "carrot": 51,
        "hot dog": 52,
        "pizza": 53,
        "donut": 54,
        "cake": 55,
        "chair": 56,
        "couch ": 57,
        "potted plant": 58,
        "bed": 59,
        "dining table": 60,
        "toilet": 61,
        "tv": 62,
        "laptop": 63,
        "mouse": 64,
        "remote": 65,
        "keyboard": 66,
        "cell phone": 67,
        "microwave": 68,
        "oven": 69,
        "toaster": 70,
        "sink": 71,
        "refrigerato": 72,
        "book": 73,
        "clock": 74,
        "vase": 75,
        "scissors": 76,
        "teddy bear": 77,
        "hair drier": 78,
        "toothbrush": 79,
    }
]


class VideoPlayer:
    def __init__(self, window, window_title, video_source):
        self.window = window
        self.window.title(window_title)

        # بارگذاری ویدیو
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            print("Error: Could not open video.")
            exit()

        # تنظیمات Canvas برای نمایش ویدیو
        self.canvas = tk.Canvas(
            window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        self.canvas.pack()

        # متغیر برای نگه‌داشتن ارجاع به تصویر
        self.photo = None

        # شروع به نمایش ویدیو
        self.update()
        self.window.mainloop()

    def update(self):
        # خواندن فریم
        ret, frame = self.vid.read()

        if ret:
            # تبدیل BGR به RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(image=frame_image)

            # به‌روزرسانی تصویر در Canvas
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # ادامه به روزرسانی فریم
        self.window.after(10, self.update)


def output(address):
    return address


def Number_Text_Captcha():
    print("please enter your captcha image address")
    address = input()
    exm = easyocr.Reader(["en"], gpu=True)
    output = exm.readtext(address, detail=0, paragraph=True, decoder="beamsearch")
    print(output)


def Image_Detail_Captcha():
    print("please enter your object who you want to detection")
    y = input()
    while True:
        if y in object_class[0]:
            main_class = object_class[0][y]
            break

        else:
            print("please enter another object or exit")
            y = input()
            if y == "exit":
                exit()

    print("please enter your captcha image address")
    address = input()
    file_name = os.path.basename(address)
    model = parse_opt()
    model.source = address
    model.weights = "yolov5m.pt"
    model.classes = main_class
    p = output(main(model))
    if p.endswith(".jpg") or p.endswith(".png") or p.endswith(".jpeg"):
        img = Image.open(p)
        return img.show()

    elif p.endswith(".mp4"):
        video_path = p  # مسیر فایل ویدئو

        root = tk.Tk()  # اجرای برنامه
        VideoPlayer(root, "Video Player", video_path)


print("Welcome to captcha solver")
print("which captcha do do you want to solve? ")
print("1- Number-text Captcha   2- image-detail Captcha ")
x = int(input())

if x == 1:
    Number_Text_Captcha()
elif x == 2:
    Image_Detail_Captcha()


# train

"""
imgs =[]
xmls =[]

train_path = 'dataset/train/images'
val_path = 'dataset/val/images'
source_path = 'dataset/main/all_files_together'

if not os.path.exists(train_path):
  os.mkdir(train_path)
if not os.path.exists(val_path):
  os.mkdir(val_path)

train_ratio = 0.8
val_ratio = 0.2

#total count of imgs
totalImgCount = len(os.listdir(source_path))/2

#sorting files to corresponding arrays
for (dirname, dirs, files) in os.walk(source_path):
    for filename in files:
        if filename.endswith('.txt'):
            xmls.append(filename)
        else:
            imgs.append(filename)


countForTrain = int(len(imgs)*train_ratio)
countForVal = int(len(imgs)*val_ratio)
print("training images are : ",countForTrain)
print("Validation images are : ",countForVal)


train_path = 'dataset/train/images'
val_path = 'dataset/val/images'
source_path = 'dataset/main/all_files_together'




#sorting files to corresponding arrays
for (dirname, dirs, files) in os.walk('dataset/labels/train'):
    for filename in files:
      print(filename)
      os.remove(os.path.join(dirname,filename))




trainimagePath = 'dataset/train/images'
trainlabelPath =  'dataset/train/labels'
valimagePath = 'dataset/val/images'
vallabelPath = 'dataset/val/labels'

if not os.path.exists(trainimagePath):
  os.mkdir(trainimagePath)
if not os.path.exists(trainlabelPath):
  os.mkdir(trainlabelPath)
if not os.path.exists(valimagePath):
  os.mkdir(valimagePath)
if not os.path.exists(vallabelPath):
  os.mkdir(vallabelPath)

for x in range(countForTrain):

    fileJpg = choice(imgs) # get name of random image from origin dir
    fileXml = fileJpg[:-4] +'.txt' # get name of corresponding annotation file


    shutil.copy(os.path.join(source_path, fileJpg), os.path.join(trainimagePath, fileJpg))
    shutil.copy(os.path.join(source_path, fileXml), os.path.join(trainlabelPath, fileXml))


    #remove files from arrays
    imgs.remove(fileJpg)
    xmls.remove(fileXml)


#cycle for test dir   
for x in range(countForVal):

    fileJpg = choice(imgs) # get name of random image from origin dir
    fileXml = fileJpg[:-4] +'.txt' # get name of corresponding annotation file

    #move both files into train dir

    shutil.copy(os.path.join(source_path, fileJpg), os.path.join(valimagePath, fileJpg))
    shutil.copy(os.path.join(source_path, fileXml), os.path.join(vallabelPath, fileXml))
    
    #remove files from arrays
    imgs.remove(fileJpg)
    xmls.remove(fileXml)

     

shutil.copy('dataset/main/dataset_yaml/dataset.yaml', 'data/dataset.yaml')

"""
