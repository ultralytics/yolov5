import os
from csv_utils import load_csv
import cv2
import shutil
import glob

def convert_to_txt(fname_to_shapes, class_to_id):
    # fname to txt rows
    # each row is: class, x_center, y_center, width, height
    # weird x-y coordinates: 
    #   x goes up from left to right
    #   y goes up from top to bottom
    fname_to_rows = {}
    for fname in fname_to_shapes:
        shapes = fname_to_shapes[fname]
        rows = []
        for shape in shapes:
            #get the image H,W
            I = cv2.imread(shape.fullpath)
            H,W = I.shape[:2]
            
            #get bbox w,h (x,y are flipped in YOLO)
            x0,y0 = shape.up_left
            x1,y1 = shape.bottom_right
            w = x1 - x0 + 1
            h = y1 - y0 + 1
            #get bbox center
            cx,cy = (x0+x1)/2, (y0+y1)/2
            # normalize to [0-1]
            class_id = class_to_id[shape.category]
            row = [class_id, cx/W, cy/H, w/W, h/H]
            rows.append(row)
        txt_name = fname.replace('.png','.txt')
        fname_to_rows[txt_name] = rows
    return fname_to_rows


def write_txts(fname_to_rows, path_txt):
    os.makedirs(path_txt, exist_ok=True)
    for fname in fname_to_rows:
        txt_file = os.path.join(path_txt, fname)
        print('writting to {}'.format(fname))
        with open(txt_file, 'w') as f:
            for class_id, cx, cy, w, h in fname_to_rows[fname]:
                row2 = '{} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(class_id, cx, cy, w, h)
                f.write(row2)


def copy_images_in_folder(path_img, path_out):
    os.makedirs(path_out, exist_ok=True)
    l = glob.glob(os.path.join(path_img, '*.png'))
    for f in l:
        shutil.copy(f, path_out)


if __name__ =='__main__':
    class_to_id = {'peeling':0, 'scuff':1, 'white':2} #0-indexed
    path_img = './data/2022-01-04_640'
    csv_file = os.path.join(path_img, 'labels.csv')
    path_out = './data/2022-01-04_640_yolo'

    if os.path.isdir(path_out):
        shutil.rmtree(path_out)

    fname_to_shapes = load_csv(csv_file, path_img)
    fname_to_rows = convert_to_txt(fname_to_shapes, class_to_id)

    path_txt = os.path.join(path_out, 'labels')
    write_txts(fname_to_rows, path_txt)

    path_img_out = os.path.join(path_out, 'images')
    copy_images_in_folder(path_img, path_img_out)
