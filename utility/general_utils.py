#!/usr/bin/env python
# -*- coding:utf-8 -*-

""" # Updated in 20/09/01 """
import coloredlogs
import logging
import configparser
import datetime
import random
import operator
import glob
import unicodedata
import os
import sys
import json
import numpy as np
import cv2
import traceback

from copy import deepcopy
from PIL import Image
from operator import itemgetter
from matplotlib import pyplot as plt
from logging import handlers as log_handlers

'''
try:
    if sys.platform == 'darwin':
       """
        import matplotlib
        matplotlib.use('TkAgg')
except all as e:
    print(e)
try:
    if sys.platform == 'darwin':
        import tkinter
        root = tkinter.Tk()
        screen_width  = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        """
       screen_width = 1440
       screen_height = 900
    else:
        from screeninfo import get_monitors
        screen_width  = get_monitors()[0].width
        screen_height = get_monitors()[0].height
    screen_width = 1920 if screen_width > 1920 else screen_width
except Exception as e:
    print('exception for screen info : {}'.format(e))
    screen_width = 1920
    screen_height = 1080
    print(" @ Warning in getting screen width and height...\n")
'''

if sys.platform == "darwin":
    screen_width = 1440
    screen_height = 900
else:
    screen_width = 1920
    screen_height = 1080

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "gif", "tif", "tiff"]
PDF_EXTENSIONS = ["pdf"]
VIDEO_EXTENSIONS = ["mp4", "avi", "mkv"]
AUDIO_EXTENSIONS = ["mp3"]
META_EXTENSION = ["json"]
IMG_EXTENSIONS = IMAGE_EXTENSIONS
LATEX_EXTENSIONS = ["tex"]
TEXT_EXTENSIONS = ["txt"]

COLOR_ARRAY_RGBCMY = [RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW]
DEV_NULL = open(os.devnull, "w")
COLORS = COLOR_ARRAY_RGBCMY


class LoggerWrapper:
    def debug(self):
        pass

    def verbose(self):
        pass

    def warning(self):
        pass

    def info(self):
        pass

    def error(self):
        pass


def read_pdf(pdf_filename, resolution=300):
    """
    Read pdf file.
    :param pdf_filename:
    :param resolution:
    :return img:
    """
    img_filename = "temp.bmp"
    convert_pdf_to_img(pdf_filename, "bmp", img_filename, resolution=resolution)
    img = imread(img_filename, color_fmt="RGB")
    os.remove(img_filename)
    return img


def convert_pdf_to_img(pdf_filename, img_type, img_filename, resolution=300):
    """
    Convert pdf file to image file.
    :param pdf_filename:
    :param img_type:
    :param img_filename:
    :param resolution:
    :return img_filename:
    """
    if os.name == "nt":
        print(" @ Error: Wand library does not work in Windows OS\n")
        sys.exit()
    else:
        from wand.image import Image

        with Image(filename=pdf_filename, resolution=resolution) as img:
            img.compression = "no"
            with img.convert(img_type) as converted:
                converted.save(filename=img_filename)
        return img_filename


def is_string_nothing(string):
    """
    check string value - nothing or not
    :param string:
    :return:
    """
    if string == "" or string is None:
        return True
    else:
        return False


def imread(img_file, color_fmt="RGB"):
    """
    Read image file.
    Support gif and pdf format.
    :param  img_file:
    :param  color_fmt: RGB, BGR, or GRAY. The default is RGB.
    :return img:
    """
    if isinstance(img_file, str):
        pass
    elif isinstance(img_file, np.ndarray):  # not isinstance(img_file, str):
        # print(" % Warning: input is NOT a string for image filename")
        # 이 경우는 img_file 이 파일 이름이 아니고 numpy array 일 경우 img_file 을 return 하는 기능이다.
        # 따라서 None 을 return 하지 말고 img_file 이 numpy array 인지를 check 하도록 수정하는 것이 좋다.
        # if 구성의 completeness를 위해 string 도 아니고 numpy array 도 아닌 경우에는 None 을 return 하도록 추가했다.
        # return None
        return img_file
    else:
        return None

    if not os.path.exists(img_file):
        print(" @ Error: image file not found {}".format(img_file))
        return None

    if not (color_fmt == "RGB" or color_fmt == "BGR" or color_fmt == "GRAY"):
        color_fmt = "RGB"

    if img_file.split(".")[-1] == "gif":
        gif = cv2.VideoCapture(img_file)
        ret, img = gif.read()
        if not ret:
            return None
    elif img_file.split(".")[-1] == "pdf":
        img = read_pdf(img_file, resolution=300)
    else:
        # img = cv2.imread(img_file.encode('utf-8'))
        # img = cv2.imread(img_file)
        # img = np.array(Image.open(img_file.encode('utf-8')).convert('RGB'), np.uint8)
        img = np.array(Image.open(img_file).convert("RGB"), np.uint8)
    if img is None:
        return None

    if color_fmt.upper() == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif color_fmt.upper() == "BGR":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        return img


def imwrite(img, img_fname, color_fmt="RGB", print_=False):
    """
    Write image file.
    :param img:
    :param img_fname:
    :param  color_fmt: RGB, BGR, or GRAY. The default is RGB.
    :param print_:
    :return img:
    """
    if len(img.shape) == 2:
        color_fmt = "GRAY"

    if color_fmt == "RGB":
        tar = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif color_fmt == "GRAY":
        tar = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif color_fmt == "BGR":
        tar = img[:]
    else:
        print(" @ Error: color_fmt, {}, is not correct.".format(color_fmt))
        return False

    temp_dir = os.path.dirname(img_fname)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    try:
        cv2.imwrite(img_fname, tar)
        if print_:
            print(" > Save image, {}.".format(img_fname))
    except Exception as ae:
        print("imwrite failed : {}".format(ae))
        return False
    return True


def imresize(img, width=None, height=None, interpolation=cv2.INTER_CUBIC):
    """
    Resize image.
    :param img:
    :param width:
    :param height:
    :param interpolation:
    :return:
    """
    if img is None or img is False:
        return None

    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape

    if width in (None, 0) and height in (None, 0):
        return img

    if width == w and height == h:
        return img

    if width in (None, 0):
        width = int(w * height / float(h))
    elif height in (None, 0):
        height = int(h * width / float(w))
    else:
        pass

    return cv2.resize(img, (width, height), interpolation)


def imresize_to_screen(img, factor=1.0):
    """
    Resize to screen size.
    :param img:
    :param factor:
    :return:
    """
    global screen_width, screen_height

    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape

    if h <= 0 or w <= 0:
        zoom_w, zoom_h = 1, 1
    else:
        zoom_w = screen_width / float(w) * 9 / 10.0 * factor
        zoom_h = screen_height / float(h) * 9 / 10.0 * factor
    zoom = min(zoom_h, zoom_w)

    return cv2.resize(img, (0, 0), fx=zoom, fy=zoom), zoom


def read_all_images(img_dir, prefix="", exts=None, color_fmt="RGB"):
    """
    Read all images with specific filename prefix in an image directory.
    :param img_dir:
    :param prefix:
    :param exts:
    :param color_fmt:
    :color_fmt:
    :return imgs: image list
    """
    if exts is None:
        exts = IMAGE_EXTENSIONS

    imgs = []

    filenames = os.listdir(img_dir)
    filenames.sort()

    for filename in filenames:
        if filename.startswith(prefix) and os.path.splitext(filename)[-1][1:] in exts:
            img = imread(os.path.join(img_dir, filename), color_fmt=color_fmt)
            imgs.append(img)

    if not imgs:
        print(' @ Error: no image filename rting with "{}"...'.format(prefix))
        sys.exit()

    return imgs


def imread_all_images(img_path, fname_prefix="", img_extensions=None, color_fmt="RGB"):
    """
    Read all images in the specific folder.
    :param img_path:
    :param fname_prefix:
    :param img_extensions:
    :param color_fmt:
    :return imgs: image list
    """
    if img_extensions is None:
        img_extensions = IMG_EXTENSIONS

    img_filenames = []
    imgs = []

    if os.path.isfile(img_path):
        filenames = [img_path]
    elif os.path.isdir(img_path):
        filenames = [os.path.join(img_path, x) for x in os.listdir(img_path)]
    else:
        print(" @ Error: The input argument is NOT a file nor folder.\n")
        return [], []

    filenames.sort()
    for filename in filenames:
        if os.path.splitext(filename)[1][1:] in img_extensions:
            if os.path.basename(filename).startswith(fname_prefix):
                imgs.append(imread(filename, color_fmt=color_fmt))
                img_filenames.append(filename)

    return imgs, img_filenames


def imshow(
    img, desc="imshow", zoom=1.0, color_fmt="RGB", skip=False, pause_sec=0, loc=(64, 64)
):
    """
    Show image.
    :param img:
    :param desc:
    :param zoom:
    :param color_fmt:
    :param skip:
    :param pause_sec:
    :param loc:
    :return:
    """
    global screen_width, screen_height
    pause_sec = -1 if pause_sec is None else pause_sec
    if skip or pause_sec < 0:
        return

    if isinstance(img, str):
        img = imread(img)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and color_fmt == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # def get_full_zoom_factor(img, zoom=0.0, factor=4/5., skip_=False):

    dim = img.shape
    h, w = dim[0], dim[1]
    zoom_w = screen_width / float(w) * 8 / 10.0
    zoom_h = screen_height / float(h) * 8 / 10.0

    if zoom == 0:
        zoom = min(zoom_h, zoom_w)
    elif (screen_width < w * zoom) or (screen_height < h * zoom):
        zoom = min(zoom_h, zoom_w)
    else:
        pass

    resize_img = cv2.resize(img, (0, 0), fx=zoom, fy=zoom)

    if os.environ.get("XSERVER"):
        if os.environ["XSERVER"] == "False":
            imwrite(
                resize_img,
                os.path.join(os.environ["HOME"], "Download", "imwrite.jpg"),
                color_fmt="BGR",
            )
            print("imwrite instead of imshow")
            return

    try:
        cv2.namedWindow(desc)
        cv2.imshow(desc, resize_img)
        if loc[0] >= 0 and loc[1] >= 0:
            cv2.moveWindow(desc, loc[0], loc[1])
        if pause_sec < 0:
            cv2.waitKey(0)
        else:
            cv2.waitKey(int(pause_sec * 1000))
        # cv2.waitKey(1)
        cv2.destroyWindow(desc)
        for k in range(10):
            cv2.waitKey(1)
    except Exception as err:
        print(err)
        plt.axis("off")
        plt.imshow(resize_img)
        if pause_sec == 0:
            plt.show()
        else:
            plt.draw()
            plt.pause(pause_sec)
        plt.close()

    return True


def crop_roi(img, roi, imshow_sec=-1):
    """
    Crop image by roi area.
    :param img:
    :param roi:
    :param imshow_sec:
    :return:
    """
    roi_mod = roi[:]
    if roi_mod[2][0] > roi_mod[3][0]:
        roi_mod[2], roi_mod[3] = roi_mod[3], roi_mod[2]

    cnr_mod = [
        [min(roi_mod[0][0], roi_mod[2][0]), min(roi_mod[0][1], roi_mod[1][1])],
        [max(roi_mod[1][0], roi_mod[3][0]), max(roi_mod[2][1], roi_mod[3][1])],
    ]
    roi_mod = [
        [cnr_mod[0][0] - roi_mod[0][0], cnr_mod[0][1] - roi_mod[0][1]],
        [cnr_mod[1][0] - roi_mod[1][0], cnr_mod[0][1] - roi_mod[1][1]],
        [cnr_mod[0][0] - roi_mod[2][0], cnr_mod[1][1] - roi_mod[2][1]],
        [cnr_mod[1][0] - roi_mod[3][0], cnr_mod[1][1] - roi_mod[3][1]],
    ]

    crop_img = img[cnr_mod[0][1] : cnr_mod[1][1], cnr_mod[0][0] : cnr_mod[1][0]]
    if imshow_sec >= 0:
        hstack_img = hstack_images(
            (
                draw_quadrilateral_on_image(
                    img.copy(), roi, color=RED, thickness=4, clockwise_=False
                ),
                crop_img,
            )
        )
        imshow(
            hstack_img,
            desc="crop_img: {} -> {}".format(img.shape[1::-1], crop_img.shape[1::-1]),
            pause_sec=imshow_sec,
        )

    # roi_corners = np.array([[tuple(x) for x in roi_mod]],dtype=np.int32)
    # ignore_mask_color = (255,) * crop_img.shape[2]
    # mask = cv2.fillPoly(np.zeros(crop_img.shape, dtype=np.uint8), roi_corners, color=ignore_mask_color)
    # crop_roi_img = cv2.bitwise_and(crop_img, mask)
    # utils.imshow(crop_roi_img, desc="crop_roi_img", pause_sec=imshow_sec)

    return crop_img, cnr_mod[0], roi_mod


def draw_line_by_equation(img, slope, sect, color=RED, thickness=3):
    """
    Draw line by equation.
    :param img:
    :param slope:
    :param sect:
    :param color:
    :param thickness:
    :return:
    """
    dim = img.shape
    pts = []
    if slope == np.inf:
        pts.append([int(sect), 0])
        pts.append([int(sect), dim[0] - 1])
    if slope == 0:
        pts.append([0, int(sect)])
        pts.append([dim[1] - 1, int(sect)])
    else:
        top_sect = int(-sect / slope)
        bot_sect = int(dim[0] - sect / slope)
        left_sect = int(sect)
        right_sect = int(slope * dim[1] + sect)
        if 0 < top_sect < dim[1]:
            pts.append([top_sect, 0])
        if 0 < bot_sect < dim[1]:
            pts.append([bot_sect, dim[1] - 1])
        if 0 < left_sect < dim[0]:
            pts.append([0, left_sect])
        if 0 < right_sect < dim[0]:
            pts.append([dim[0] - 1, right_sect])
        if len(pts) != 2:
            print(" @ Error : pts length is not 2")
            sys.exit()

    img = cv2.line(img, tuple(pts[0]), tuple(pts[1]), color=color, thickness=thickness)

    return img


def draw_box_on_img(img, box, color=RED, thickness=2, alpha=0.5):
    """
    Draw a box overlay to image.
    box format is either 2 or 4 vertex points.
    :param img:
    :param box:
    :param color:
    :param thickness:
    :param alpha:
    :return:
    """
    if np.array(box).size == 8:
        box = [box[0][0], box[0][1], box[3][0], box[3][1]]

    box = [int(v) for v in box]
    box_img = cv2.rectangle(
        deepcopy(img), tuple(box[0:2]), tuple(box[2:]), color, thickness
    )
    box_img = cv2.addWeighted(img, alpha, box_img, 1 - alpha, 0)
    return box_img


def draw_boxes_on_img(
    img, boxes, color=RED, thickness=2, alpha=0.0, margin=0, add_cross_=False
):
    """
    Draw the overlay of boxes to an image.
    box format is either 2 or 4 vertex points.
    :param img:
    :param boxes:
    :param color: color vector such as (R,G,B) or (R,G,B,alpha) or string 'random'
    :param thickness:
    :param alpha:
    :param margin:
    :param add_cross_:
    :return:
    """
    margins = [x * margin for x in [-1, -1, 1, 1]]

    if isinstance(color, str):
        if color.lower() == "random":
            box_color = -1
        else:
            box_color = RED
        box_alpha = alpha
    else:
        if len(color) == 4:
            box_color = color[:3]
            box_alpha = color[3]
        else:
            box_color = color
            box_alpha = alpha

    box_img = np.copy(img)
    for cnt, box in enumerate(boxes):
        if np.array(box).size == 8:
            box = [box[0][0], box[0][1], box[3][0], box[3][1]]
        if box_color == -1:
            rand_num = random.randint(0, len(COLORS) - 1)
            mod_color = COLORS[rand_num]
        else:
            mod_color = box_color
        mod_box = list(map(operator.add, box, margins))
        box_img = cv2.rectangle(
            box_img, tuple(mod_box[0:2]), tuple(mod_box[2:]), mod_color, thickness
        )
        if add_cross_:
            box_img = cv2.line(
                box_img,
                (mod_box[0], mod_box[1]),
                (mod_box[2], mod_box[3]),
                color=BLACK,
                thickness=8,
            )
            box_img = cv2.line(
                box_img,
                (mod_box[2], mod_box[1]),
                (mod_box[0], mod_box[3]),
                color=BLACK,
                thickness=8,
            )
    disp_img = cv2.addWeighted(np.copy(img), box_alpha, box_img, 1 - box_alpha, 0)

    return disp_img


def add_box_overlay(img, box, color, alpha):
    """
    Add overlay box to image.
    :param img:
    :param box:
    :param color:
    :param alpha:
    :return:
    """
    over = cv2.rectangle(img.copy(), tuple(box[0:2]), tuple(box[2:]), color, -1)
    over = cv2.addWeighted(img.copy(), alpha, over, 1 - alpha, 0)
    return over


def get_center_of_mass_from_quad(quad):
    x, y = 0, 0
    tot_num = float(len(quad))
    for pnt in quad:
        x += pnt[0]
        y += pnt[1]
    return x / tot_num, y / tot_num


def draw_quadrilateral_on_image(
    img, quad_arr, color=RED, thickness=2, clockwise_=False
):
    """
    Draw a quadrilateral on image.
    This function includes the regularization of quadrilateral vertices.
    :param img:
    :param quad_arr:
    :param color:
    :param thickness:
    :param clockwise_:
    :return:
    """
    mod_quad_arr = deepcopy(quad_arr)
    disp_img = np.copy(img)
    if mod_quad_arr is None:
        return disp_img
    if not mod_quad_arr:
        return disp_img
    if len(np.array(mod_quad_arr).shape) != 3:
        mod_quad_arr = [mod_quad_arr]

    for quad in mod_quad_arr:

        """
        if quad:
            if isinstance(quad[0], list):
                mod_quad = np.array(regularize_quadrilateral_vertices(quad), dtype=np.int32)
            else:
                mod_quad = [[quad[0], quad[1]], [quad[2], quad[1]],
                            [quad[0], quad[3]], [quad[2], quad[3]]]
            if clockwise_:
                mod_quad[2], mod_quad[3] = mod_quad[3], mod_quad[2]
            disp_img = cv2.line(disp_img, tuple(mod_quad[0]), tuple(mod_quad[1]), color=color, thickness=thickness)
            disp_img = cv2.line(disp_img, tuple(mod_quad[0]), tuple(mod_quad[2]), color=color, thickness=thickness)
            disp_img = cv2.line(disp_img, tuple(mod_quad[3]), tuple(mod_quad[1]), color=color, thickness=thickness)
            disp_img = cv2.line(disp_img, tuple(mod_quad[3]), tuple(mod_quad[2]), color=color, thickness=thickness)
        """
        if not isinstance(quad[0], list):
            mod_quad = [
                [quad[0], quad[1]],
                [quad[2], quad[1]],
                [quad[0], quad[3]],
                [quad[2], quad[3]],
            ]
        else:
            mod_quad = quad
        if clockwise_:
            mod_quad[2], mod_quad[3] = mod_quad[3], mod_quad[2]

        disp_img = cv2.line(
            disp_img,
            tuple(mod_quad[0]),
            tuple(mod_quad[1]),
            color=color,
            thickness=thickness,
        )
        disp_img = cv2.line(
            disp_img,
            tuple(mod_quad[0]),
            tuple(mod_quad[2]),
            color=color,
            thickness=thickness,
        )
        disp_img = cv2.line(
            disp_img,
            tuple(mod_quad[3]),
            tuple(mod_quad[1]),
            color=color,
            thickness=thickness,
        )
        disp_img = cv2.line(
            disp_img,
            tuple(mod_quad[3]),
            tuple(mod_quad[2]),
            color=color,
            thickness=thickness,
        )

    return disp_img


def generate_four_vertices_from_ref_vertex(ref, img_sz):
    """
    Generate four vertices from top-left reference vertex.
    :param ref:
    :param img_sz:
    :return:
    """
    pt_tl = [int(img_sz[0] * ref[0]), int(img_sz[1] * ref[1])]
    # pt_tr = [int(img_sz[0]), pt_tl[1]]
    pt_tr = [int(img_sz[0] - pt_tl[0]), pt_tl[1]]
    pt_bl = [pt_tl[0], int(img_sz[1] - pt_tl[1])]
    pt_br = [pt_tr[0], pt_bl[1]]

    return [pt_tl, pt_tr, pt_bl, pt_br]


def crop_image_from_ref_vertex(img, ref_vertex, symm_crop_=True, debug_=False):
    """
    Crop input image with reference vertex.
    :param img:
    :param ref_vertex:
    :param symm_crop_:
    :param debug_:
    :return:
    """
    pts = generate_four_vertices_from_ref_vertex(ref_vertex, img.shape[1::-1])
    if symm_crop_:
        crop_img = img[pts[0][1] : pts[3][1], pts[0][0] : pts[1][0]]
    else:
        crop_img = img[pts[0][1] : pts[3][1], pts[0][0] : img.shape[1]]

    if debug_:
        imshow(
            draw_box_on_img(img, pts, color=RED, thickness=10, alpha=0.5),
            desc="original image with frame",
        )
        imshow(crop_img, desc="cropped image")
    return crop_img


def crop_image_by_corners(
    img, corners, corner_type="zigzag", method="perspective", debug_=False
):
    """
    Crop input image with 2 or 4 corners.
    :param img:
    :param corners:
    :param corner_type: zigzag, counterclockwise
    :param method: perspective, max_area
    :param debug_:
    :return:
    """
    if np.array(corners).shape == (2, 2):
        return img[corners[0][1] : corners[1][1], corners[0][0] : corners[1][0]]

    if method == "perspective":

        if corner_type == "zigzag":
            src_pts = np.float32([corners[0], corners[1], corners[3], corners[2]])
        elif corner_type == "counterclockwise":
            src_pts = np.float32(corners)
        else:
            print(" @ Error: incorrect corner_type, {}.")
            return None
        # tar_pts = np.float32(get_corners2_from_corners4(src_pts, corner_type=corner_type, method='average', margin=0))
        tar_pts = np.float32(
            get_corners2_from_corners4(src_pts, method="average", margin=0)
        )
        tar_pts = np.float32(
            [
                tar_pts[0],
                [tar_pts[1][0], tar_pts[0][1]],
                tar_pts[1],
                [tar_pts[0][0], tar_pts[1][1]],
            ]
        )
        tar_pts -= tar_pts[0]

        matrix = cv2.getPerspectiveTransform(src_pts, tar_pts)
        crop_img = cv2.warpPerspective(img, matrix, (tar_pts[2][0], tar_pts[2][1]))

        if debug_:
            cnr_img = draw_quadrilateral_on_image(
                np.copy(img), corners, color=RED, thickness=4
            )
            imshow(cnr_img, "image with four corners")
            imshow(crop_img, desc="cropped image")

        return crop_img

    elif method == "max_area":
        return None
    else:
        print(" @ Error: incorrect method, {}".format(method))
        return None


def crop_image_with_coordinates(img, crop_coordinates):
    """
    Crop image by coordinates.
    :param img:
    :param crop_coordinates:
    :return:
    """
    width_point_start = int(img.shape[1] * crop_coordinates[0])
    width_point_end = int(img.shape[1] * crop_coordinates[1])
    height_point_rt = int(img.shape[0] * crop_coordinates[2])
    height_point_end = int(img.shape[0] * crop_coordinates[3])
    crop_img = img[height_point_rt:height_point_end, width_point_start:width_point_end]

    return crop_img


def get_datetime(fmt="%Y-%m-%d_%H:%M:%S.%f"):
    """
    Get datetime with format argument.
    :param fmt:
    :return:
    """
    return datetime.datetime.now().strftime(fmt)


def setup_logger_with_ini(ini, logging_=True, console_=True):
    backup_count = 0
    if "backup_count" in ini:
        backup_count = int(ini["backup_count"])

    logger = setup_logger(
        ini["name"],
        ini["prefix"],
        folder=ini["folder"],
        filename=None,  # 'test.log',
        backup_count=backup_count,
        logger_=logging_,
        console_=console_,
    )

    return logger


def setup_logger(
    logger_name,
    log_prefix_name,
    level=logging.INFO,
    folder=".",
    filename=None,
    backup_count=0,
    logger_=True,
    console_=True,
):
    """Setup logger supporting two handlers of stdout and file.
    :param logger_name:
    :param log_prefix_name:
    :param level:
    :param folder:
    :param filename:
    :param backup_count:
    :param logger_:
    :param console_:
    :return:
    """
    if not logger_:
        logger = LoggerWrapper()
        logger.debug = print
        logger.verbose = print
        logger.info = print
        logger.error = print
        return logger

    if not os.path.exists(folder):
        os.makedirs(folder)

    CUSTOM_LEVEL_STYLES = dict(
        spam=dict(color="green", faint=True),
        debug=dict(color="green"),
        verbose=dict(color="blue"),
        info=dict(color="cyan"),
        notice=dict(color="magenta"),
        warning=dict(color="yellow"),
        success=dict(color="green", bold=True),
        error=dict(color="red"),
        critical=dict(color="red", bold=True),
    )

    # 로거 파일은 사용자가 지정하는 고정적인 파일 이름 대신 date 와 time 으로 구성하는 것이 자동 로거를 구현할때 일반적이다.
    # 이전에 test.log 를 로거 파일 이름으로 설정한 이유는 단순히 테스트를 위한 것이다.
    # 아래에 filename 이 있을 경우와 없을 경우를 나누어서 로거의 파일 이름을 설정하도록 변경한다.
    dt = get_datetime()[:-2].replace(":", "-")
    log_file = (
        os.path.join(*folder.split("/"), log_prefix_name + dt + ".log")
        if filename is None
        else filename
    )
    log_setup = logging.getLogger(logger_name)
    formatter = coloredlogs.ColoredFormatter(
        "%(name)-10s | %(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level_styles=CUSTOM_LEVEL_STYLES,
    )
    # formatter = logging.Formatter('%(name)-10s | %(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s',
    #                               datefmt='%Y-%m-%d %H:%M:%S')

    # log 파일이 무한정 커지는 것을 방지하고, 서버 시작후 부터 일자별로 최대 backup_count(기본=0) 일까지 저장하도록 한다.
    if backup_count > 0:
        file_handler = log_handlers.TimedRotatingFileHandler(
            log_file, backupCount=backup_count, when="midnight"
        )
    else:
        file_handler = logging.FileHandler(log_file, mode="a")

    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y%m%d.bak"
    stream_handler = None
    if console_:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(file_handler)
    if console_ and stream_handler:
        log_setup.addHandler(stream_handler)
    return logging.getLogger(logger_name)


def get_stdout_logger(logger=None):
    if logger is None:
        logger = LoggerWrapper()
        logger.info = print
        logger.error = print
    return logger


def folder_exists(in_dir, type_="rst", exit_=False, create_=False, print_=False):
    """
    Check if a directory exists or not. If not, create it according to input argument.
    :param in_dir:
    :param exit_:
    :param create_:
    :param print_:
    :return:
    """
    if not in_dir:
        return

    if os.path.isdir(in_dir):
        if print_:
            print(" # Info: directory, {}, already existed.".format(in_dir))
        return True
    else:
        if create_:
            try:
                # print(in_dir)
                os.makedirs(in_dir)
            except OSError:
                print(" @ Error: make_dirs in check_directory_existence routine...\n")
                sys.exit()
        else:
            if print_:
                print("\n @ Warning: directory not found, {}.\n".format(in_dir))
            if exit_:
                sys.exit()
        return False


def file_exists(filename, print_=False, exit_=False):
    """
    Check if a file exists or not.
    :param filename:
    :param print_:
    :param exit_:
    :return True/False:
    """
    if not os.path.isfile(filename):
        if print_ or exit_:
            print("\n @ Warning: file not found, {}.\n".format(filename))
        if exit_:
            sys.exit()
        return False
    else:
        return True


def get_filenames_in_a_directory(dir_name):
    """
    Get names of all the files in a directory.
    :param dir_name:
    :return out_filenames:
    """
    filenames = os.listdir(dir_name)
    out_filenames = []
    for filename in filenames:
        if os.path.isfile(os.path.join(dir_name, filename)):
            out_filenames.append(filename)

    return out_filenames


def transpose_list(in_list):
    """
    Transpose a 2D list variable.
    :param in_list:
    :return:
    """
    try:
        len(in_list[0])
        return list(map(list, zip(*in_list)))
    except TypeError:
        return in_list


def plt_imshow(
    data_2d,
    title=None,
    x_label=None,
    y_label=None,
    x_range=None,
    y_range=None,
    xticks=None,
    yticks=None,
    maximize_=True,
    block_=True,
):
    """
    Show image via matplotlib.pyplot.
    :param data_2d:
    :param title:
    :param x_label:
    :param y_label:
    :param x_range:
    :param y_range:
    :param xticks:
    :param yticks:
    :param maximize_:
    :param block_:
    :return:
    """

    maximize_ = maximize_ and False
    if maximize_:
        if os.name == "nt":  # If Windows OS.
            plt.get_current_fig_manager().window.state("zoomed")
        else:
            plt.get_current_fig_manager().window.showMaximized()

    dim = data_2d.shape
    if len(dim) == 2:
        plt.imshow(data_2d, cmap="gray")
    elif len(dim) == 3:
        if dim[2] == 1:
            plt.imshow(data_2d, cmap="gray")
        else:
            plt.imshow(data_2d)

    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)
    plt.xticks(xticks), plt.yticks(yticks)
    plt.show(block=block_)


def check_string_in_class(class_name, sub_string):
    for attr in dir(class_name):
        if sub_string in attr:
            print(attr)


def vstack_images(imgs, margin=20):
    """
    ck images vertically with boundary and in-between margin.
    :param imgs:
    :param margin:
    :return:
    """
    widths = []
    heights = []
    num_imgs = len(imgs)

    if num_imgs == 1:
        return imgs[0]

    color_images = []
    for img in imgs:
        img_sz = img.shape[1::-1]
        widths.append(img_sz[0])
        heights.append(img_sz[1])
        color_images.append(img)

    max_width = max(widths) + 2 * margin
    max_height = sum(heights) + (num_imgs + 1) * margin
    if len(imgs[0].shape) == 3:
        vck_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    else:
        vck_image = np.zeros((max_height, max_width), dtype=np.uint8)

    x_offset = margin
    y_offset = margin
    for img in color_images:
        img_sz = img.shape[1::-1]
        vck_image[
            y_offset : y_offset + img_sz[1], x_offset : x_offset + img_sz[0]
        ] = img
        y_offset += margin + img_sz[1]

    return vck_image


def hstack_images(imgs, margin=20):
    """
    ck images horizontally with boundary and in-between margin.
    :param imgs:
    :param margin:
    :return:
    """
    widths = []
    heights = []
    num_imgs = len(imgs)

    if num_imgs == 1:
        return imgs[0]

    color_images = []
    for img in imgs:
        img_sz = img.shape[1::-1]
        widths.append(img_sz[0])
        heights.append(img_sz[1])
        color_images.append(img)

    max_width = sum(widths) + (num_imgs + 1) * margin
    max_height = max(heights) + 2 * margin
    if len(imgs[0].shape) == 3:
        hck_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    else:
        hck_image = np.zeros((max_height, max_width), dtype=np.uint8)

    x_offset = margin
    y_offset = margin
    for img in color_images:
        img_sz = img.shape[1::-1]
        hck_image[
            y_offset : y_offset + img_sz[1], x_offset : x_offset + img_sz[0]
        ] = img
        x_offset += margin + img_sz[0]

    return hck_image


def get_filenames(
    dir_path, prefixes=("",), extensions=("",), recursive_=False, exit_=False
):
    """
    Find all the files rting with prefixes or ending with extensions in the directory path.
    ${dir_path} argument can accept file.
    :param dir_path:
    :param prefixes:
    :param extensions:
    :param recursive_:
    :param exit_:
    :return:
    """
    if os.path.isfile(dir_path):
        return [dir_path]

    if not os.path.isdir(dir_path):
        return []

    dir_name = os.path.dirname(dir_path)

    filenames = glob.glob(dir_name + "**/**", recursive=recursive_)
    for i in range(len(filenames) - 1, -1, -1):
        basename = os.path.basename(filenames[i])
        if not (
            os.path.isfile(filenames[i])
            and basename.startswith(tuple(prefixes))
            and basename.endswith(tuple(extensions))
        ):
            del filenames[i]

    if len(filenames) == 0:
        print(" @ Error: no file detected in {}".format(dir_path))
        if exit_:
            sys.exit(1)

    return filenames


"""
    for path, subdirs, files in os.walk(dir_path):
        for name in files:
            if not name.startswith(".DS"):
                if name.startswith(tuple(prefixes)):
                    filenames.append(os.path.join(path, name))
                if name.endswith(tuple(extensions)):
                    filenames.append(os.path.join(path, name))
    if (not filenames) and (not extensions) and (not prefixes):
        filenames = glob.glob(dir_path + "*")
    if len(filenames) == 0:
        print(" @ Error: no file detected in {}".format(dir_path))
        if exit_:
            sys.exit(1)
"""


def check_box_boundary(box, sz):
    box[0] = 0 if box[0] < 0 else box[0]
    box[1] = 0 if box[1] < 0 else box[1]
    box[2] = sz[0] if box[2] > sz[0] else box[2]
    box[3] = sz[1] if box[3] > sz[1] else box[3]
    return box


def regularize_quadrilateral_vertices(vertices):
    """
    Regularize quadrilateral vertices to the de-facto rule. (LT, RT, LB, RB)
    :param vertices: 2D list of position list of (x, y).
    :return:
    """
    out = sorted(vertices, key=itemgetter(0))
    if out[0][1] > out[1][1]:
        out[0], out[1] = out[1], out[0]
    if out[2][1] > out[3][1]:
        out[2], out[3] = out[3], out[2]
    out[1], out[2] = out[2], out[1]
    return out


def get_bool_from_ini(ini_param):
    """
    Get boolean value from M$ INI style configuration.
    :param ini_param:
    :return: True, False, or None.
    """
    if not isinstance(ini_param, str):
        return None
    if ini_param.lower() in ["0", "off", "false"]:
        return False
    elif ini_param.lower() in ["1", "on", "true"]:
        return True
    else:
        return None


def remove_comments_in_ini_section(ini_section, cmt_string="###"):
    """
    Remove comments in ini section.
    where comment is the sentence rting the special string combination such as '###'
    :param ini_section:
    :param cmt_string:
    :return:
    """
    ini_out = ini_section
    for key in ini_section:
        ini_out[key] = ini_section[key].split(cmt_string)[0]
    return ini_out


def remove_comments_in_ini(ini, cmt_delimiter="###"):
    """
    Remove comments in ini file,
    where comment is text strings rting with comment delimiter.
    :param ini:
    :param cmt_delimiter:
    :return:
    """
    for section in ini.sections():
        for key in ini[section]:
            ini[section][key] = ini[section][key].split(cmt_delimiter)[0].strip()
    return ini


def split_fname(fname):
    """
    Split the filename into folder, core name, and extension.
    :param fname:
    :return:
    """
    folder = os.path.dirname(fname)
    base_fname = os.path.basename(fname)
    split = os.path.splitext(base_fname)
    core_fname = split[0]
    ext = split[1]
    return folder, core_fname, ext


def copy_folder_structure(src_path, tar_path):
    """
    Copy structure of folder
    :param src_path:
    :param tar_path:
    :return:
    """
    dir_names = glob.glob(src_path + "**/*/", recursive=True)
    for dir_name in dir_names:
        tar_name = os.path.join(tar_path, dir_name[len(src_path) :])
        print(tar_name)
        if not os.path.isdir(tar_name):
            os.makedirs(tar_name)
    return True


"""    
    for dir_path, dir_names, file_names in os.walk(src_path):
        structure = os.path.join(tar_path, dir_path[len(src_path):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print(" % Info : {} folder does already exist...".format(structure))
    return True
"""


class JsonConvert(object):
    mappings = {}

    @classmethod
    def class_mapper(cls, d):
        for keys, cl in cls.mappings.items():
            if keys.issuperset(d.keys()):  # are all required arguments present?
                return cl(**d)
        else:
            # Raise exception instead of silently returning None
            raise ValueError(
                "Unable to find a matching class for object: {!s}".format(d)
            )

    @classmethod
    def complex_handler(cls, obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            raise TypeError(
                "Object of type %s with value of %s is not JSON serializable"
                % (type(obj), repr(obj))
            )

    @classmethod
    def register(cls, in_class):
        cls.mappings[
            frozenset(tuple([attr for attr, val in cls().__dict__.items()]))
        ] = in_class
        return cls

    @classmethod
    def to_json(cls, obj):
        return json.dumps(obj.__dict__, default=cls.complex_handler, indent=4)

    @classmethod
    def from_json(cls, json_str):
        return json.loads(json_str, object_hook=cls.class_mapper)

    @classmethod
    def to_file(cls, obj, path):
        with open(path, "w") as f:
            f.writelines([cls.to_json(obj)])
        return path

    @classmethod
    def from_file(cls, filepath):
        with open(filepath, "r") as f:
            result = cls.from_json(f.read())
        return result


def get_color(i=None, primary_=False):
    if primary_:
        color_array = COLOR_ARRAY_RGBCMY[:3]
    else:
        color_array = COLOR_ARRAY_RGBCMY
    if i is None:
        return color_array[random.randint(0, len(color_array) - 1)]
    else:
        return color_array[i % len(color_array)]


def is_json(my_json):
    try:
        json.loads(my_json)
        return True
    except ValueError:
        return False


def transform_quadrilateral_to_rectangle(quad, algo="max", margin=16):
    """
    Find an appropriate rectangle from quadrilateral.
    NEXT:
    Need to implement an algorithm to transform an severe skewed quadrilateral to rectangle.
    :param quad:
    :param algo: algorithm to transform input quadrilateral to rectangle, ['max'].
    :param margin:
    :return:
    """
    if np.array(quad).shape != (4, 2):
        return None
    if algo == "max":
        t = [list(np.amin(quad, axis=0)), list(np.amax(quad, axis=0))]
        x1 = t[0][0] - margin
        y1 = t[0][1] - margin
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        return [[x1, y1], [t[1][0] + margin, t[1][1] + margin]]
    else:
        return None


def unicode_normalize(string):
    if string is None:
        return string
    else:
        return unicodedata.normalize("NFC", string)


def compare_image_folders(folder1, folder2, direction="horizontal", pause_sec=0):
    """
    Compare image folders
    :param folder1:
    :param folder2:
    :param direction:
    :param pause_sec:
    :return:
    """
    filenames1 = sorted(get_filenames_in_a_directory(folder1))
    for filename in filenames1:
        if os.path.splitext(filename)[1][1:] in IMAGE_EXTENSIONS:
            print(" # Compare {}".format(filename))
            img1 = imread(os.path.join(folder1, filename))
            if os.path.isfile(os.path.join(folder2, filename)):
                img2 = imread(os.path.join(folder2, filename))
            else:
                img2 = np.zeros(tuple(img1.shape), dtype=np.uint8)
            if direction == "horizontal":
                img = hstack_images((img1, img2))
            else:
                img = vstack_images((img1, img2))
            imshow(
                img,
                desc="compare image " + os.path.basename(filename),
                pause_sec=pause_sec,
            )


def check_range(val, min_val, max_val):
    if val < min_val:
        return min_val
    elif val > max_val:
        return max_val
    else:
        return val


def recv_all(con, recv_buf_size=1024, timeout_val=60.0, logger=None):
    """
    Receive files.
    :param con:
    :param recv_buf_size:
    :param timeout_val:
    :param logger:
    :return:
    """
    byte_data = b""
    data_len_list = None  # [16, 15, 527, 837, 842]
    while True:
        try:
            con.settimeout(timeout_val)
            part = con.recv(recv_buf_size)
            if logger:
                # logger.info(" #recv_all# ({:d} : {}".format(len(part), str(part)))
                pass
            if len(part) > 0:
                byte_data += part
                if is_json(byte_data):
                    break
                if data_len_list:
                    try:
                        if data_len_list.index(len(byte_data)) >= 0:
                            if logger:
                                # logger.info(" #recv_all# Total packet length is {:d}".format(len(byte_data)))
                                pass
                            return byte_data
                    except ValueError:
                        pass
            else:
                break
        except Exception as err:
            if logger:
                logger.error(str(err) + "\n" + traceback.format_exc())
            break

    return byte_data


def get_ini_parameters(ini_fname, cmt_delimiter="###"):
    """
    Initialize parameters.
    :param ini_fname:
    :param cmt_delimiter:
    :return:
    """
    ini = configparser.ConfigParser()
    file_exists(ini_fname, exit_=True)
    ini.read(ini_fname, encoding="utf-8")
    return remove_comments_in_ini(ini, cmt_delimiter=cmt_delimiter)


def get_rect_size_from_quad(quad):
    """
    Find the size of quad image.
    :param quad:
    :return:
    """
    sz_x = (abs(quad[1][0] - quad[0][0]) + abs(quad[3][0] - quad[2][0])) / 2.0
    sz_y = (abs(quad[2][1] - quad[0][1]) + abs(quad[3][1] - quad[1][1])) / 2.0
    return sz_x, sz_y


def get_corners2_from_corners4(corner, method="maximum", margin=0, flatten_=False):
    """
    Convert 2 corners info into 4 corners (coordinates).
    :param corner:
    :param method: maximum or average.
    :param margin:
    :param flatten_:
    :return:
    """

    mod_cnr4 = deepcopy(corner)
    if mod_cnr4[2][0] > mod_cnr4[3][0]:
        tmp = deepcopy(mod_cnr4[2])
        mod_cnr4[2] = mod_cnr4[3]
        mod_cnr4[3] = tmp

    rect = []
    if method == "maximum":
        rect = [
            [
                min(mod_cnr4[0][0], mod_cnr4[2][0]) - margin,
                min(mod_cnr4[0][1], mod_cnr4[1][1]) - margin,
            ],
            [
                max(mod_cnr4[1][0], mod_cnr4[3][0]) + margin,
                max(mod_cnr4[2][1], mod_cnr4[3][1]) + margin,
            ],
        ]
    elif method == "average":
        rect = [
            [
                (mod_cnr4[0][0] + mod_cnr4[2][0]) / 2 - margin,
                (mod_cnr4[0][1] + mod_cnr4[1][1]) / 2 - margin,
            ],
            [
                (mod_cnr4[1][0] + mod_cnr4[3][0]) / 2 + margin,
                (mod_cnr4[2][1] + mod_cnr4[3][1]) / 2 + margin,
            ],
        ]
    else:
        print(
            " @ Error in get_corners2_from_corners4 : incorrect method argument, {}.".format(
                method
            )
        )

    if flatten_:
        rect = [rect[0][0], rect[0][1], rect[1][0], rect[1][1]]

    return rect


def update_image_arr_for_video(img_arr):
    """
    Update image array in order to make video with image array
    which might have different shape from each other.
    :param img_arr:
    :return updated_img_arr:
    """
    if not img_arr:
        return None

    max_width = max([img.shape[1] for img in img_arr])
    max_height = max([img.shape[0] for img in img_arr])
    updated_img_arr = []
    for img in img_arr:
        img_shape = img.shape
        max_img = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        max_img[0 : img_shape[0], 0 : img_shape[1]] = img
        updated_img_arr.append(max_img)
    return updated_img_arr


def save_dict_to_json_file(dict_dat, json_fname, logger=get_stdout_logger()):
    """
    Save dictionary in json format
    :param dict_dat:
    :param json_fname:
    :param logger:
    :return:
    """
    with open(json_fname, "w") as f:
        json.dump(dict_dat, f)
    if logger:
        logger.info()


def get_all_files_in_dir_path():
    pass


def check_file_existence():
    pass


def get_random_color():
    pass


def check_directory_existence():
    pass


get_all_files_in_dir_path = get_filenames
get_random_color = get_color
check_file_existence = file_exists
check_directory_existence = folder_exists
COLOR_ARRAY = COLOR_ARRAY_RGBCMY
