"""
convert the data (images with a csv annotation file) to yolo file format
"""

#built-in packages
import os
import cv2
import shutil
import glob
import json

#LMI utils packages
#need to activate this environment: lmi_ai.env
from csv_utils import load_csv

def convert_to_txt(fname_to_shapes, class_to_id):
    """
    convert the map <fname, list of shape objects> to YOLO format
    Arguments:
        fname_to_shapes(dict): the map <fname, list of shape objects>
        class_to_id(dict): the map <class_name, class_ID>
    Return:
        fname_to_rows: the map <file name, list of row>, where each row is [class_ID, x, y, w, h]
    """
    fname_to_rows = {}
    for fname in fname_to_shapes:
        shapes = fname_to_shapes[fname]
        rows = []
        for shape in shapes:
            #get class ID
            class_id = class_to_id[shape.category]
            #get the image H,W
            I = cv2.imread(shape.fullpath)
            H,W = I.shape[:2]
            #get bbox w,h
            x0,y0 = shape.up_left
            x1,y1 = shape.bottom_right
            w = x1 - x0 + 1
            h = y1 - y0 + 1
            #get bbox center
            cx,cy = (x0+x1)/2, (y0+y1)/2
            # normalize to [0-1]
            row = [class_id, cx/W, cy/H, w/W, h/H]
            rows.append(row)
        txt_name = fname.replace('.png','.txt')
        fname_to_rows[txt_name] = rows
    return fname_to_rows


def write_txts(fname_to_rows, path_txts):
    """
    write to the yolo format txts
    Arugments:
        fname_to_rows(dict): a map <filename, a list of rows>, where each row is [class_ID, x, y, w, h]
        path_txts: the output folder contains txt files
    """
    os.makedirs(path_txts, exist_ok=True)
    for fname in fname_to_rows:
        txt_file = os.path.join(path_txts, fname)
        print('writting to {}'.format(fname))
        with open(txt_file, 'w') as f:
            for class_id, cx, cy, w, h in fname_to_rows[fname]:
                row2 = '{} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(class_id, cx, cy, w, h)
                f.write(row2)


def copy_images_in_folder(path_img, path_out):
    """
    copy the images from one folder to another
    Arguments:
        path_img(str): the path of original image folder
        path_out(str): the path of output folder
    """
    os.makedirs(path_out, exist_ok=True)
    l = glob.glob(os.path.join(path_img, '*.png'))
    for f in l:
        shutil.copy(f, path_out)


if __name__ =='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--path_imgs', type=str, required=True, help='the path of a data folder, where it contains images and a labels.csv file')
    ap.add_argument('-o', '--path_out', type=str, required=True, help='the output path')
    args = vars(ap.parse_args())

    path_imgs = args['path_imgs']
    csv_file = os.path.join(path_imgs, 'labels.csv')

    if not os.path.isfile(csv_file):
        raise Exception(f'cannot find labels.csv in {path_imgs}')

    fname_to_shapes,class_to_id = load_csv(csv_file, path_imgs, zero_index=True)
    fname_to_rows = convert_to_txt(fname_to_shapes, class_to_id)

    #generate output yolo dataset
    if os.path.isdir(args['path_out']):
        print('found output folder, deleting...')
        shutil.rmtree(args['path_out'])
    os.makedirs(args['path_out'])

    #write class map file
    fname = os.path.join(args['path_out'], 'class_map.json')
    with open(fname, 'w') as outfile:
        json.dump(class_to_id, outfile)

    #write labels/annotations
    path_txts = os.path.join(args['path_out'], 'labels')
    write_txts(fname_to_rows, path_txts)

    #write images
    path_img_out = os.path.join(args['path_out'], 'images')
    copy_images_in_folder(path_imgs, path_img_out)
