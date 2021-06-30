#!/usr/bin/env python
# -*- coding:utf-8 -*-

""" # Updated in 20/06/24 """

import os
import time
import cv2, math
import numpy as np
import operator
import imutils
import operator
from PIL import Image, ImageDraw, ImageFont

from utility import general_utils as utils

HANGUL_FONT = "../common/fonts/gothic.ttf"


def split_area_by_condition(
    img,
    roi_ratio=0.2,
    enable_=True,
    mode="line_detection",
    line_detection_algorithm="HoughLineTransform",
    check_time_=False,
    pause_sec=-1,
    save_img_=False,
    save_fpath=None,
    logger=utils.get_stdout_logger(),
):
    start_time = time.time() if check_time_ else None

    if enable_ == False:
        return [img], False

    img_h, img_w, img_c = img.shape
    center_x = int(img_w / 2)
    x_margin = center_x * roi_ratio

    # ROI 추출(라인 검출 영역)
    roi = [
        int(center_x - x_margin),
        0,
        int(center_x + x_margin),
        img_h,
    ]  # x, y, x+w, x+h
    crop_img = img[roi[1] : roi[3], roi[0] : roi[2]]

    # 영역 좌표 계산
    split_ = False
    area_pts = []
    if mode == "contour_detection":
        split_, contour_pts = check_contours_in_img(
            crop_img, pause_sec=pause_sec, save_img_=save_img_, save_fpath=save_fpath
        )
        area_pts = contour_pts

    elif mode == "line_detection":
        split_, line_pts = check_lines_in_img(
            crop_img,
            line_detection_algorithm,
            get_line_=True,
            pause_sec=pause_sec,
            save_img_=save_img_,
            save_fpath=save_fpath,
        )
        area_pts = line_pts

    if split_ is False:
        logger.info(" [SPLIT-AREA] # boundary lines not detected.")
        if check_time_:
            logger.info(
                " [SPLIT-AREA] # elapsed time : {:.3f} sec".format(
                    float(time.time() - start_time)
                )
            )
        return [img], split_
    else:
        logger.info(
            " [SPLIT-AREA] # {:d} boundary lines detected.".format(len(area_pts))
        )
        trans_line_pts = utils.transpose_list(area_pts)
        start_xs, start_ys, end_xs, end_ys = (
            trans_line_pts[0],
            trans_line_pts[1],
            trans_line_pts[2],
            trans_line_pts[3],
        )
        line_xs = start_xs + end_xs
        avg_line_x = sum(line_xs) // len(line_xs)

        # 원본 좌표로 변환
        split_x = int(avg_line_x + roi[0])
        split_y = int(start_ys[0] + roi[1])

        # 이미지 영역 분할
        split_row_imgs = [img[:split_y, :], img[split_y:, :]]
        top_img, bottom_img = split_row_imgs
        split_imgs = [top_img] + [bottom_img[:, :split_x], bottom_img[:, split_x:]]
        if check_time_:
            logger.info(
                " [SPLIT-AREA] # elapsed time : {:.3f} sec".format(
                    float(time.time() - start_time)
                )
            )
        return split_imgs, split_


def check_contours_in_img(img, pause_sec=-1, save_img_=False, save_fpath=None):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_contour = np.copy(img)
    else:
        img_gray = np.copy(img)
        img_contour = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    ret, img_bw = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY)

    # Specify size on vertical axis
    height, width, channel = img.shape
    vertical_size = height // 30

    # Create structure element for extracting vertical lines through morphology operations
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size - 6))

    # Apply morphology operations
    img_v = cv2.dilate(img_bw, kernel)
    img_v = cv2.erode(img_v, vertical_kernel)
    img_v = cv2.erode(img_v, vertical_kernel)

    img_v[img_v < 255] = 0

    # Find countours
    _, contours, _ = cv2.findContours(img_v, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Remove unnecessary contour boxes
    contour_pts = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        box_size = w * h
        max_box_size = height * width * (1 / 3)
        box_h = h
        min_height = height * (1 / 5)
        if box_size > max_box_size:
            continue
        elif box_h < min_height:
            continue
        contour_pts.append([x, y, x + w, y + h])

    # Select the largest box height
    if len(contour_pts) > 0:
        contour_pt = max((pt for pt in contour_pts), key=lambda x: (x[3] - x[1]))
        rect_x, rect_y, rect_w, rect_h = contour_pt
        cv2.rectangle(img_contour, (rect_x, rect_y), (rect_w, rect_h), (0, 0, 255), 2)

    utils.imshow(img_contour, pause_sec=pause_sec, desc="contour detection")
    if save_img_:
        dir_name, file_name, _ = utils.split_fname(save_fpath)
        contour_img_fname = os.path.join(
            dir_name, "split_area", "contour_" + file_name + ".jpg"
        )
        utils.imwrite(img_contour, contour_img_fname)

    if len(contour_pts) > 0:
        return True, contour_pts
    else:
        return False, contour_pts


def check_lines_in_img(
    img,
    algorithm="HoughLineTransform",
    get_line_=False,
    pause_sec=-1,
    save_img_=False,
    save_dir_name="./",
    save_fpath=None,
):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb = np.copy(img)
    else:
        img_gray = np.copy(img)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    ret, img_bw = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY)

    # Specify size on vertical axis
    rows = img.shape[0]
    verticalsize = rows // 30

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = cv2.dilate(img_bw, verticalStructure)
    vertical = cv2.erode(vertical, verticalStructure)

    # Inverse pixel value
    img_inv = cv2.bitwise_not(vertical)

    line_pts = []
    if algorithm == "HoughLineTransform":
        lines = cv2.HoughLines(img_inv, 1, np.pi / 180, 100)
        print(" # Total lines: {:d}".format(len(lines)))
        for line in lines:
            img_lines = np.copy(img_rgb)
            dim = img_lines.shape
            rho = line[0][0]
            theta = line[0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x = []
            y = []
            if b != 0:
                slope = -a / b
                y1 = slope * (-x0) + y0
                if 0 <= y1 < dim[0]:
                    x.append(0)
                    y.append(y1)
                y1 = slope * (dim[1] - 1 - x0) + y0
                if 0 <= y1 < dim[0]:
                    x.append(dim[1] - 1)
                    y.append(y1)
                x1 = (-y0) / slope + x0
                if 0 <= x1 < dim[1]:
                    x.append(x1)
                    y.append(0)
                x1 = (dim[0] - 1 - y0) / slope + x0
                if 0 <= x1 < dim[1]:
                    x.append(x1)
                    y.append(dim[0] - 1)
            else:
                x = [x0, x0]
                y = [0, dim[0] - 1]
            angle = 90 - (theta * 180 / np.pi)
            print(" # Rotated angle = {:.1f} <- ({:f}, {:f})".format(angle, theta, rho))
            if len(x) == 2:
                img_lines = cv2.line(
                    img_rgb,
                    (int(x[0]), int(y[0])),
                    (int(x[1]), int(y[1])),
                    utils.RED,
                    4,
                )
                # utils.imshow(img_lines)
                start_pos = (int(x[0]), int(y[0]))
                end_pos = (int(x[1]), int(y[1]))
                line_pts.append(start_pos, end_pos)
            else:
                print(" @ Warning: something wrong.\n")
                pass

    elif algorithm == "ProbabilisticHoughTransform":
        dim = img_inv.shape
        height = dim[0]
        min_line_length = height // 8
        max_line_gap = min_line_length
        linesP = cv2.HoughLinesP(
            img_inv,
            1,
            np.pi / 180,
            threshold=100,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        img_lines = np.copy(img_rgb)
        if linesP is None:
            print(" # Total lines: 0")
            if get_line_:
                return False, []
            else:
                return False
        else:
            num_linesP = len(linesP)
            print(" # Total lines: {}".format(num_linesP))
            for line in linesP:
                cv2.line(
                    img_lines, tuple(line[0][0:2]), tuple(line[0][2:]), utils.RED, 10
                )  # (x0, y0), (x1, y1)
                # angle = np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0]) * 180. / np.pi
                # print(" # Rotated angle = {:.1f}".format(angle))
                line_pts.append(line.tolist()[0])
    utils.imshow(img_lines, pause_sec=pause_sec, desc="line detection")
    if save_img_:
        dir_name, file_name, _ = utils.split_fname(save_fpath)
        line_img_fname = os.path.join(
            dir_name, "split_area", "line_" + file_name + ".jpg"
        )
        utils.imwrite(img_lines, line_img_fname)

    if get_line_:
        return True, line_pts
    else:
        return True


def draw_line_from_rho_and_theta(img, rho, theta, pause_sec=-1):
    img_sz = img.shape[1::-1]
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    x = []
    y = []
    if b != 0:
        slope = -a / b
        y1 = slope * (-x0) + y0
        if 0 <= y1 < img_sz[1]:
            x.append(0)
            y.append(y1)
        y1 = slope * (img_sz[0] - 1 - x0) + y0
        if 0 <= y1 < img_sz[1]:
            x.append(img_sz[0] - 1)
            y.append(y1)
        x1 = (-y0) / slope + x0
        if 0 <= x1 < img_sz[0]:
            x.append(x1)
            y.append(0)
        x1 = (img_sz[1] - 1 - y0) / slope + x0
        if 0 <= x1 < img_sz[0]:
            x.append(x1)
            y.append(img_sz[1] - 1)
    else:
        x = [x0, x0]
        y = [0, img_sz[1] - 1]
    angle = 90 - (theta * 180 / np.pi)
    if pause_sec >= 0:
        print(" # Rotated angle = {:f} <- ({:.3f}, {:.3f})".format(angle, theta, rho))
    if len(x) == 2:
        pts = [[int(x[0] + 0.5), int(y[0] + 0.5)], [int(x[1] + 0.5), int(y[1] + 0.5)]]
    else:
        if pause_sec >= 0:
            print(" @ Warning: rho is zero.\n")
        pts = [[0, 0], [0, 0]]

    line_img = np.copy(img)
    cv2.line(line_img, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), utils.RED, 4)
    utils.imshow(line_img, pause_sec=pause_sec)

    return pts


def derotate_image(
    img,
    inside_margin_ratio=20,
    max_angle=30,
    max_angle_candidates=50,
    angle_resolution=0.5,
    check_time_=False,
    pause_sec=-1,
    save_img_=False,
    save_fpath=None,
    logger=utils.get_stdout_logger(),
):
    start_time = time.time() if check_time_ else None
    img_gray = (
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else np.copy(img)
    )

    try:
        inside_margin = [int(x * inside_margin_ratio) for x in img.shape[1::-1]]
    except Exception as e:
        logger.info(" @ OCR Error : {}".format(e))
        return img, 0

    img_gray[: inside_margin[1], :] = 255
    img_gray[-inside_margin[1] :, :] = 255
    img_gray[:, : inside_margin[0]] = 255
    img_gray[:, -inside_margin[0] :] = 255

    # noinspection PyUnreachableCode
    if False:
        check_lines_in_img(img, algorithm="HoughLineTransform")
        check_lines_in_img(img, algorithm="ProbabilisticHoughTransform")

    ret, img_bw = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )  # defualt min : 128
    """
    kernel = np.ones((5, 5), np.uint8)  # note this is a horizontal kernel
    bw = np.copy(img_bw)
    for i in range(9):
        bw = cv2.erode(bw, kernel, iterations=1)
        bw = cv2.dilate(bw, kernel, iterations=1)
    utils.imshow(utils.hstack_images((img_bw, bw)))
    """
    img_edge = cv2.Canny(img_bw, 50, 150, apertureSize=3)

    # noinspection PyUnreachableCode
    if False:
        utils.imshow(img_edge)

    lines = cv2.HoughLines(img_edge, 1, np.pi / 360, int(min(img_edge.shape) / 8.0))

    angles = []
    if lines is not None:
        for cnt, line in enumerate(lines):
            angle = (
                int((90 - line[0][1] * 180 / np.pi) / float(angle_resolution))
                * angle_resolution
            )
            draw_line_from_rho_and_theta(img, line[0][0], line[0][1], pause_sec=-1)

            if abs(angle) < max_angle:
                angles.append(angle)
            if max_angle_candidates < cnt:
                break

    # rot_angle = max(set(angles), key=angles.count)
    sorted_angles = sorted(
        {x: angles.count(x) for x in angles}.items(),
        key=operator.itemgetter(1),
        reverse=True,
    )

    if len(sorted_angles) == 0:
        rot_angle = 0
    elif len(sorted_angles) == 1:
        rot_angle = sorted_angles[0][0]
    elif sorted_angles[0][0] == 0 and (sorted_angles[0][1] < 2 * sorted_angles[1][1]):
        rot_angle = sorted_angles[1][0]
    elif (sorted_angles[0][1] / sorted_angles[1][1]) < 3 and abs(
        sorted_angles[0][0] - sorted_angles[1][0]
    ) <= 1.0:
        rot_angle = (sorted_angles[0][0] + sorted_angles[1][0]) / 2.0
    else:
        rot_angle = sorted_angles[0][0]

    """
    if rot_angle != 0:
        rot_angle += 0.5
    """
    logger.info(" [DEROTATION] # {:.1f} degree detected.".format(rot_angle))

    sz = img_bw.shape[1::-1]
    rot_img = ~imutils.rotate(
        ~img, angle=-rot_angle, center=(int(sz[0] / 2), int(sz[1] / 2)), scale=1
    )

    if check_time_:
        logger.info(
            " [DEROTATION] # elapsed time : {:.3f} sec".format(
                float(time.time() - start_time)
            )
        )

    utils.imshow(
        np.concatenate((img, rot_img), axis=1), pause_sec=pause_sec, desc="de-rotation"
    )

    if save_img_:
        dir_name, file_name, _ = utils.split_fname(save_fpath)
        rot_img_fname = os.path.join(
            dir_name, "derotate", "derot_" + file_name + ".jpg"
        )
        utils.imwrite(np.concatenate((img, rot_img), axis=1), rot_img_fname)

    return rot_img, rot_angle


def transform_bboxes(bboxes, angle, src_shape, dst_shape):
    ori_bboxes = []
    for box in bboxes:
        ori_bbox = []
        for i in range(len(box)):
            ori_bbox.append(
                transform_point(box[i], np.radians(angle), src_shape, dst_shape)
            )
        ori_bboxes.append(ori_bbox)
    return ori_bboxes


def transform_point(src_point, radian, src_shape, dst_shape):
    """Transform the point from source to destination image.
    Args:
        src_point: Point coordinates in source image.
        angle: Rotate angle.
        src_shape: The shape of source image.
        dst_shape: The shape of destination image.
    Returns:
        List of new point coordinates in rotated image.
    """
    x, y = src_point
    src_offset_x, src_offset_y = src_shape[1] / 2.0, src_shape[0] / 2.0

    adjusted_x = x - src_offset_x
    adjusted_y = y - src_offset_y
    cos_rad = math.cos(radian)
    sin_rad = math.sin(radian)
    qx = int(src_offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y)
    qy = int(src_offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y)

    dst_offset_x, dst_offset_y = (dst_shape[1] - src_shape[1]) / 2.0, (
        dst_shape[0] - src_shape[0]
    ) / 2.0
    dst_x, dst_y = int(qx + dst_offset_x), int(qy + dst_offset_y)
    return [dst_x, dst_y]
