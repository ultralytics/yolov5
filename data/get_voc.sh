# PASCAL VOC dataset http://host.robots.ox.ac.uk/pascal/VOC/
# Download command: bash yolov5/data/get_voc.sh
# Train command: python train.py --data voc.yaml
# Dataset should be placed next to yolov5 folder:
#   /parent_folder
#     /VOC
#     /yolov5

start=`date +%s`

# handle optional download dir
if [ -z "$1" ]
  then
    # navigate to ~/data
    echo "navigating to ../data/ ..." 
    mkdir -p ../data
    cd ../data/
  else
    # check if is valid directory
    if [ ! -d $1 ]; then
        echo $1 "is not a valid directory"
        exit 0
    fi
    echo "navigating to" $1 "..."
    cd $1
fi

echo "Downloading VOC2007 trainval ..."
# Download the data.
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
echo "Downloading VOC2007 test data ..."
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
echo "Done downloading."

# Extract data
echo "Extracting trainval ..."
tar -xf VOCtrainval_06-Nov-2007.tar
echo "Extracting test ..."
tar -xf VOCtest_06-Nov-2007.tar
echo "removing tars ..."
rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"

start=`date +%s`

# handle optional download dir
if [ -z "$1" ]
  then
    # navigate to ~/data
    echo "navigating to ../data/ ..." 
    mkdir -p ../data
    cd ../data/
  else
    # check if is valid directory
    if [ ! -d $1 ]; then
        echo $1 "is not a valid directory"
        exit 0
    fi
    echo "navigating to" $1 "..."
    cd $1
fi

echo "Downloading VOC2012 trainval ..."
# Download the data.
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
echo "Done downloading."


# Extract data
echo "Extracting trainval ..."
tar -xf VOCtrainval_11-May-2012.tar
echo "removing tar ..."
rm VOCtrainval_11-May-2012.tar

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"

cd ../data
echo "Spliting dataset..."
python3 - "$@" <<END
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        convert_annotation(year, image_id)
    list_file.close()

END

cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt
cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt

python3 - "$@" <<END

import shutil
import os
os.system('mkdir ../VOC/')
os.system('mkdir ../VOC/images')
os.system('mkdir ../VOC/images/train')
os.system('mkdir ../VOC/images/val')

os.system('mkdir ../VOC/labels')
os.system('mkdir ../VOC/labels/train')
os.system('mkdir ../VOC/labels/val')

import os
print(os.path.exists('../data/train.txt'))
f = open('../data/train.txt', 'r')
lines = f.readlines()

for line in lines:
    #print(line.split('/')[-1][:-1])
    line = "/".join(line.split('/')[2:])
    #print(line)
    if (os.path.exists("../" + line[:-1])):
        os.system("cp ../"+ line[:-1] + " ../VOC/images/train")
        
print(os.path.exists('../data/train.txt'))
f = open('../data/train.txt', 'r')
lines = f.readlines()

for line in lines:
    #print(line.split('/')[-1][:-1])
    line = "/".join(line.split('/')[2:])
    line = line.replace('JPEGImages', 'labels')
    line = line.replace('jpg', 'txt')
    #print(line)
    if (os.path.exists("../" + line[:-1])):
        os.system("cp ../"+ line[:-1] + " ../VOC/labels/train")

print(os.path.exists('../data/2007_test.txt'))
f = open('../data/2007_test.txt', 'r')
lines = f.readlines()

for line in lines:
    #print(line.split('/')[-1][:-1])
    line = "/".join(line.split('/')[2:])
    
    if (os.path.exists("../" + line[:-1])):
        os.system("cp ../"+ line[:-1] + " ../VOC/images/val")

print(os.path.exists('../data/2007_test.txt'))
f = open('../data/2007_test.txt', 'r')
lines = f.readlines()

for line in lines:
    #print(line.split('/')[-1][:-1])
    line = "/".join(line.split('/')[2:])
    line = line.replace('JPEGImages', 'labels')
    line = line.replace('jpg', 'txt')
    #print(line)
    if (os.path.exists("../" + line[:-1])):
        os.system("cp ../"+ line[:-1] + " ../VOC/labels/val")

END

rm -rf ../data
