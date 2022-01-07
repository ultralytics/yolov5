import cv2
import os
import glob

#import 3rd party libaries
import mask
import rect
import csv_utils


def resize_imgs_with_annotations(img_dir, csv_path, output_imsize):
    file_list = glob.glob(os.path.join(img_dir, '*.png'))
    shapes = csv_utils.load_csv(csv_path, path_img=img_dir)
    name_to_im = {}
    for file in file_list:
        im = cv2.imread(file)
        im_name = os.path.basename(file)
        input_size = im.shape[:2]
        ratio_x = output_imsize[0]/input_size[0]
        ratio_y = output_imsize[1]/input_size[1]
        assert(ratio_x==ratio_y,'asepect ratio changed')
        
        out_name = im_name.replace('w{}'.format(input_size[0]),'w{}'.format(output_imsize[0]))
        out_name = out_name.replace('h{}'.format(input_size[1]),'h{}'.format(output_imsize[1]))
        
        ratio = ratio_x
        im2 = cv2.resize(im,dsize=output_imsize)
        name_to_im[out_name] = im2

        for i in range(len(shapes[im_name])):
            if isinstance(shapes[im_name][i], rect.Rect):
                shapes[im_name][i].up_left = [int(v*ratio) for v in shapes[im_name][i].up_left]
                shapes[im_name][i].bottom_right = [int(v*ratio) for v in shapes[im_name][i].bottom_right]
                shapes[im_name][i].im_name = out_name
            elif isinstance(shapes[im_name][i], mask.Mask):
                shapes[im_name][i].X = [int(v*ratio) for v in shapes[im_name][i].X]
                shapes[im_name][i].Y = [int(v*ratio) for v in shapes[im_name][i].Y]
                shapes[im_name][i].im_name = out_name
            else:
                raise Exception("Found unsupported classes. Supported classes are mask and rect")
    return name_to_im, shapes



if __name__=='__main__':
    import argparse
    output_imsize = (640,640)
    
    img_dir = './data/allImages_1024'
    csv_path = os.path.join(img_dir, 'labels.csv')
    output_dir = f'./data/2022-01-04_{output_imsize[0]}'
    

    name_to_im,shapes = resize_imgs_with_annotations(img_dir, csv_path, output_imsize)
    # write out to disk
    os.makedirs(output_dir,exist_ok=True)
    for im_name in name_to_im:
        cv2.imwrite(os.path.join(output_dir,im_name), name_to_im[im_name])
    csv_utils.write_to_csv(shapes, os.path.join(output_dir,'labels.csv'))
