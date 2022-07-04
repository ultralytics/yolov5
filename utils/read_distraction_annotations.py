import json
import os
import random
from collections import namedtuple
from pathlib import Path
from typing import Dict, Tuple, List, Union

import cv2
import numpy as np
from absl import app
from absl import flags
from absl import logging


def get_jpg_encoded_cropped_img(
    img_path: str, crop_offset: Tuple[int, int], im_root: str = None
) -> np.ndarray:
    """Read, crop and encode an image
    Args:
        img_path: str containing path to image
        crop_offset: A tuple for cropping image pixels from x-y directions
        im_root: str containing image root directory path

    Returns:
        cropped and encoded image
    """

    r_off = crop_offset[0]
    c_off = crop_offset[1]

    im_name = img_path.split("/")[-1]
    if im_root is not None:
        img_path = im_root + im_name
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[r_off:, c_off:, :]

    img_encode = cv2.imencode(".jpg", img)[1]

    return img_encode.tobytes()


def heatmap_2_pose(
    idx: np.array, img_shape: Tuple[int, int], args: namedtuple
) -> np.ndarray:
    """convert heatmap to upper body pose coordinates
    Args:
        idx: array of indices for max location in heatmap
        img_shape: tuple of original image shape
        args: a namedtupe of configuration from config and yaml file

    Returns:
        kpts: array of upper body pose x-y coordinates
    """
    kpts = []
    mask_shape = args.pose_mask_shape
    res_shape = args.image_size

    if (
        any(idx) > 0
    ):  # kps is a numpy array of shape 7, 2 or an empty list (if gt has no kps)
        x = idx % mask_shape[1]
        y = idx // mask_shape[1]

        scale_x0 = float(res_shape[1]) / img_shape[1]
        scale_y0 = float(res_shape[0]) / img_shape[0]
        scale_x1 = float(mask_shape[1]) / res_shape[1]
        scale_y1 = float(mask_shape[0]) / res_shape[0]
        x_orig = np.int32(np.floor(x / (scale_x0 * scale_x1)))
        y_orig = np.int32(np.floor(y / (scale_y0 * scale_y1)))

        kpts = np.vstack((x_orig, y_orig)).T
    return kpts


def shift_box(box: np.array, offset: Tuple[int, int]) -> np.array:
    """shifts bboxes using given offset
    Args:
        box: array of bbox coordinates
        offset: tuple of offset coordinates

    Returns:
        bbox: updated bbox
    """
    if len(box):
        box[0] -= offset[1]
        box[2] -= offset[1]
        box[1] -= offset[0]
        box[3] -= offset[0]
    return box


def shift_kpts(kpts: np.array, offset: Tuple[int, int]) -> np.array:
    """shifts kpts using given offset
    Args:
        kpts: array of pose coordinates
        offset: tuple of offset coordinates

    Returns:
        kpts: updated kpts
    """
    if len(kpts) and any(kpts[:, 0] > 0):
        kpts[:, 0] -= offset[1]
        kpts[:, 1] -= offset[0]
    return kpts


def labels_2_hot_vector(
    labels_list: List[str], mapping: Dict[str, int], num_classification_classes: int
) -> np.ndarray:
    """converts classification labels to one hot vector
    Args:
        labels_list: list of labels
        mapping: a dict containing labels mappings

    Returns:
        hot_vec: one hot vector for given labels_list
    """
    # hot vector length is num_classification_classes + 1, the last one is for not sure and not used for training
    hot_vec = np.zeros(num_classification_classes + 1, dtype=int)
    ids = [mapping[label] - 1 for label in labels_list if label in mapping.keys()]
    hot_vec[ids] = 1
    return hot_vec


def process_pose(
    kps: np.array, img_shape: Tuple[int, int], args: namedtuple
) -> Tuple[np.array, np.array]:
    """converts pose to heatmap
    Args:
        kps: array of keypoints
        img_shape: original image shape
        args: a namedtuple of configuration from config and yaml file

    Returns:
        heatmaps: an array containing indices for keypoints locations
        weights: a bool array that contains 1 if give pose keypoint is valid (lies inside pose mask shape)
            and 0 otherwise
    """
    heatmaps = np.zeros(args['num_pose_kps'], np.int64)
    weights = np.zeros(args['num_pose_kps'], np.int64)
    mask_shape = args['pose_mask_shape']
    res_shape = args['image_size']

    if (
        len(kps) > 0
    ):  # kps is a numpy array of shape 7, 2 or an empty list (if gt has no kps)
        kps = kps.reshape((args['num_pose_kps'], 2))  # ask why it was 3
        scale_x0 = float(res_shape[1]) / img_shape[1]
        scale_y0 = float(res_shape[0]) / img_shape[0]

        scale_x1 = float(mask_shape[1]) / res_shape[1]
        scale_y1 = float(mask_shape[0]) / res_shape[0]

        x_orig = kps[:, 0].astype(np.float32)
        y_orig = kps[:, 1].astype(np.float32)

        x_boundary_inds = np.where(x_orig == img_shape[1])[0]
        y_boundary_inds = np.where(y_orig == img_shape[0])[0]

        xx = np.floor(x_orig * scale_x0 * scale_x1)
        yy = np.floor(y_orig * scale_y0 * scale_y1)

        if len(x_boundary_inds) > 0:
            xx[x_boundary_inds] = mask_shape[1] - 1

        if len(y_boundary_inds) > 0:
            yy[y_boundary_inds] = mask_shape[0] - 1

        valid_loc = np.logical_and(
            np.logical_and(xx >= 0, yy >= 0),
            np.logical_and(xx < mask_shape[1], yy < mask_shape[0]),
        )  # changed mask_shape[0] to [1]

        valid = valid_loc.astype(np.int64)
        lin_ind = (yy * mask_shape[1]) + xx  # changed mask_shape[1] to [0]
        heatmaps = lin_ind * valid
        heatmaps = heatmaps.astype(np.int64)
        weights = valid.astype(np.int64)

    return heatmaps, weights


def get_bboxes_and_labels(annotations: Dict) -> Tuple[List[list], List[str]]:
    """get and return bboxes and labels list from annotation dict
    Args:
        annotations: a dict containing annotations

    Returns:
        bboxes: list of all the bounding boxes
        bboxes_labels: list of bboxes labels
    """
    bboxes = []
    bboxes_labels = []

    if len(annotations["annotations"]["person_bbox"]):
        body_bb = [
            annotations["annotations"]["person_bbox"]["xmin"],
            annotations["annotations"]["person_bbox"]["ymin"],
            annotations["annotations"]["person_bbox"]["xmax"],
            annotations["annotations"]["person_bbox"]["ymax"],
        ]
        bboxes.append(body_bb)
        bboxes_labels.append("person")
    if len(annotations["annotations"]["face_bbox"]):
        face_bb = [
            annotations["annotations"]["face_bbox"]["xmin"],
            annotations["annotations"]["face_bbox"]["ymin"],
            annotations["annotations"]["face_bbox"]["xmax"],
            annotations["annotations"]["face_bbox"]["ymax"],
        ]
        bboxes.append(face_bb)
        bboxes_labels.append("face")
    if len(annotations["annotations"]["mobile_bb"]):
        mobile_bb = [
            annotations["annotations"]["mobile_bb"]["xmin"],
            annotations["annotations"]["mobile_bb"]["ymin"],
            annotations["annotations"]["mobile_bb"]["xmax"],
            annotations["annotations"]["mobile_bb"]["ymax"],
        ]
        bboxes.append(mobile_bb)
        bboxes_labels.append("mobile")

    # handle additional objects
    # mounted cellphone
    mounted_cp_bb = annotations["annotations"].get("mounted_cp_bbox", None)
    if isinstance(mounted_cp_bb, list) and len(mounted_cp_bb) >= 1:
        mounted_cp_bb = mounted_cp_bb[0]
    if mounted_cp_bb:
        bbox = [
            mounted_cp_bb["xmin"],
            mounted_cp_bb["ymin"],
            mounted_cp_bb["xmax"],
            mounted_cp_bb["ymax"],
        ]
        bboxes.append(bbox)
        bboxes_labels.append("mounted_cp")   

    # hands
    left_hand_bb = annotations["annotations"].get("left_hand_bb", None)
    if left_hand_bb:
        bbox = [
            left_hand_bb["xmin"],
            left_hand_bb["ymin"],
            left_hand_bb["xmax"],
            left_hand_bb["ymax"],
        ]
        bboxes.append(bbox)
        bboxes_labels.append("hand")   

    right_hand_bb = annotations["annotations"].get("right_hand_bb", None)
    if right_hand_bb:
        bbox = [
            right_hand_bb["xmin"],
            right_hand_bb["ymin"],
            right_hand_bb["xmax"],
            right_hand_bb["ymax"],
        ]
        bboxes.append(bbox)
        bboxes_labels.append("hand")   

    # hands not tagged from prism, then check for other sources like yolov5
    if not left_hand_bb and not right_hand_bb:
        hands_bb = annotations["annotations"].get("hands_bb", None)
        if hands_bb is not None:
            if isinstance(hands_bb, list) and len(hands_bb) >= 1:
                hands_bb = hands_bb[0]
            if len(hands_bb):
                for hand_bb in hands_bb:
                    bbox = [
                        hand_bb["xmin"],
                        hand_bb["ymin"],
                        hand_bb["xmax"],
                        hand_bb["ymax"],
                    ]
                    bboxes.append(bbox)
                    bboxes_labels.append("hand")

    # smoking
    smoking_bb = annotations["annotations"].get("smoking_item_bb", None)
    if smoking_bb is not None:
        if isinstance(smoking_bb, list) and len(smoking_bb) >= 1:
            smoking_bb = smoking_bb[0]
        if len(smoking_bb):
            bbox = [
                smoking_bb["xmin"],
                smoking_bb["ymin"],
                smoking_bb["xmax"],
                smoking_bb["ymax"],
            ]
            bboxes.append(bbox)
            bboxes_labels.append("smoking")

    # drinking
    drinking_bb = annotations["annotations"].get("drinking_item_bb", None)
    if drinking_bb is not None:
        if isinstance(drinking_bb, list) and len(drinking_bb) >= 1:
            drinking_bb = drinking_bb[0]
        if len(drinking_bb):
            bbox = [
                drinking_bb["xmin"],
                drinking_bb["ymin"],
                drinking_bb["xmax"],
                drinking_bb["ymax"],
            ]
            bboxes.append(bbox)
            bboxes_labels.append("drinking")

    # drinking
    food_item_bb = annotations["annotations"].get("food_item_bb", None)
    if food_item_bb is not None:
        if isinstance(food_item_bb, list) and len(food_item_bb) >= 1:
            food_item_bb = food_item_bb[0]
        if len(food_item_bb):
            bbox = [
                food_item_bb["xmin"],
                food_item_bb["ymin"],
                food_item_bb["xmax"],
                food_item_bb["ymax"],
            ]
            bboxes.append(bbox)
            bboxes_labels.append("drinking")

    # other item
    other_item_bb = annotations["annotations"].get("other_item_bb", None)
    if other_item_bb is not None:
        if isinstance(other_item_bb, list) and len(other_item_bb) >= 1:
            other_item_bb = other_item_bb[0]
        if len(other_item_bb):
            bbox = [
                other_item_bb["xmin"],
                other_item_bb["ymin"],
                other_item_bb["xmax"],
                other_item_bb["ymax"],
            ]
            bboxes.append(bbox)
            bboxes_labels.append("other_item")

    return bboxes, bboxes_labels

def xyxy2xywhn(box_xyxy, w, h):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    box_xywh = [0, 0, 0, 0]
    box_xywh[0] = ((box_xyxy[0] + box_xyxy[2]) / 2) / w  # x center
    box_xywh[1] = ((box_xyxy[1] + box_xyxy[3]) / 2) / h  # y center
    box_xywh[2] = (box_xyxy[2] - box_xyxy[0]) / w  # width
    box_xywh[3] = (box_xyxy[3] - box_xyxy[1]) / h  # height
    return box_xywh

def read_annotation_json_file(
    ann_path: str, args: namedtuple
) -> Dict[str, Union[List[list], List[str], np.ndarray, np.ndarray]]:
    """reads annotations from json file and post process raw annotations and return processed annotations dictionary
    Args:
        ann_path: a str for json annotation file
        args: a dict of configuration from config and yaml file

    Returns:
        data: a dictionary containing annotation data for
    """
    # read annotations
    with open(ann_path) as f:
        annotations = json.load(f)

    bboxes, bboxes_labels = get_bboxes_and_labels(annotations=annotations)

    distraction_tags = annotations["annotations"]["distraction_tags"]
    face_kpts = annotations["annotations"]["face_kpts"]
    face_kpts_tag = annotations["annotations"]["face_kpts_tag"]
    body_kpts = annotations["annotations"]["body_kpts"]
    body_kpts_tag = annotations["annotations"]["body_kpts_tag"]
    seat_belt_tag = annotations["annotations"]["seat_belt_tag"]
    reflective_jacket_tag = annotations["annotations"]["reflective_jacket_tag"]
    camera_obstruction_tag = annotations["annotations"]["camera_obstruction_tag"]
    orig_image_width = annotations["width"]
    orig_image_height = annotations["height"]
    image_name = annotations["ImageName"]

    # if orig_image_height > 720:
    #     crop_offset = args['crop_offset_1080p']
    # else:
    #     crop_offset = args['crop_offset_720p']

    # cropped_image_width = orig_image_width - crop_offset[1]
    # cropped_image_height = orig_image_height - crop_offset[0]

    # shift boxes
    # bboxes_shifted = []
    bboxes_cxywh = []
    for i, box in enumerate(bboxes):
        # bboxes_shifted.append(shift_box(box=box, offset=crop_offset))
        box_xywh = xyxy2xywhn(bboxes[i], orig_image_width, orig_image_height)
        c = args['label_name_map'][bboxes_labels[i]]
        bboxes_cxywh.append( [c] + box_xywh)


    # shift body pose kpts
    # if body_kpts_tag != "unacceptable" and len(body_kpts) > 0:
    #     body_kpts_shifted = shift_kpts(np.asarray(body_kpts), crop_offset)
    # else:
    #     body_kpts_shifted = []

    body_kpts_processed, body_kpts_processed_weights = process_pose(
        np.asarray(body_kpts), (orig_image_height, orig_image_width), args
    )

    normalized_kpts = [ [kpt[0]/orig_image_width, kpt[1]/orig_image_height] for kpt in body_kpts]

    # filter classification label
    # 1: Not_distracted, Looking_left/right, looking_down
    # 2: mobile_usage
    # 3: other_distraction
    # 4: properly_fastened
    # 5: not_properly_fastened, not_fastened
    # 6: not_sure
    if isinstance(distraction_tags, str):
        labels = (
            [distraction_tags]
            + [seat_belt_tag]
            + [reflective_jacket_tag]
            + [camera_obstruction_tag]
        )
    else:
        labels = (
            distraction_tags
            + [seat_belt_tag]
            + [reflective_jacket_tag]
            + [camera_obstruction_tag]
        )
    classification_labels_one_hot_vec = labels_2_hot_vector(
        labels_list=labels,
        mapping=args['ground_truth_label_map_dict'],
        num_classification_classes=args['num_classification_classes'],
    )
    # # if looking down occurs with any other tag then disable looking down
    # if ("looking_down" in labels) and ("mobile_usage" in labels or "other_distraction" in labels or "smoking" in labels or "eating" in labels or "drinking" in labels or "holding_any_object_in_hand" in labels):
    #     classification_labels_one_hot_vec[6] = 0
    # # if looiking_left-right/un-distracted occurs with any other class then disable undistracted/looking-left-right
    # if ("Looking_left/right" in labels or "Not_distracted" in labels) and ("mobile_usage" in labels or "other_distraction" in labels or "smoking" in labels or "eating" in labels or "drinking" in labels or "holding_any_object_in_hand" in labels):
    #     classification_labels_one_hot_vec[0] = 0

    data = {
        "image_name": image_name,
        "bboxes": bboxes,
        "bboxes_labels": bboxes_labels,
        "classification_labels": classification_labels_one_hot_vec,
        #"body_kpts": body_kpts_processed,
        #"body_kpts_weights": body_kpts_processed_weights,
        "bboxes_cxywh": bboxes_cxywh,
        "normalized_kpts": normalized_kpts
    }

    return data
