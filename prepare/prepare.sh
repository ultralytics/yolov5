python split_images2txt.py
echo "split images to txt"
python split_images_from_txt.py
echo "split images from txt"
python voc2txt.py
echo "connvert VOC to Txt"
python split_labels_fromOne2N.py
echo "split labels to dir"
rm -rf ../dataset/train.txt ../dataset/val.txt ../dataset/test.txt ../dataset/prelabels/
echo "delete train.txt val.txt test.txt prelabels/"
