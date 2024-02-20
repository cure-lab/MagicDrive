from typing import Tuple, Union
import os
import cv2
import copy
import logging
import tempfile
from PIL import Image

import numpy as np
import torch
from accelerate.scheduler import AcceleratedScheduler

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera


def box_center_shift(bboxes: LiDARInstance3DBoxes, new_center):
    raw_data = bboxes.tensor.numpy()
    new_bboxes = LiDARInstance3DBoxes(
        raw_data, box_dim=raw_data.shape[-1], origin=new_center)
    return new_bboxes


def trans_boxes_to_views(bboxes, transforms, aug_matrixes=None, proj=True):
    """This is a wrapper to perform projection on different `transforms`.

    Args:
        bboxes (LiDARInstance3DBoxes): bboxes
        transforms (List[np.arrray]): each is 4x4.
        aug_matrixes (List[np.array], optional): each is 4x4. Defaults to None.

    Returns:
        List[np.array]: each is Nx8x3, where z always equals to 1 or -1
    """
    if len(bboxes) == 0:
        return None

    coords = []
    for idx in range(len(transforms)):
        if aug_matrixes is not None:
            aug_matrix = aug_matrixes[idx]
        else:
            aug_matrix = None
        coords.append(
            trans_boxes_to_view(bboxes, transforms[idx], aug_matrix, proj))
    return coords


def trans_boxes_to_view(bboxes, transform, aug_matrix=None, proj=True):
    """2d projection with given transformation.

    Args:
        bboxes (LiDARInstance3DBoxes): bboxes
        transform (np.array): 4x4 matrix
        aug_matrix (np.array, optional): 4x4 matrix. Defaults to None.

    Returns:
        np.array: (N, 8, 3) normlized, where z = 1 or -1
    """
    if len(bboxes) == 0:
        return None

    bboxes_trans = box_center_shift(bboxes, (0.5, 0.5, 0.5))
    trans = transform
    if aug_matrix is not None:
        aug = aug_matrix
        trans = aug @ trans
    corners = bboxes_trans.corners
    num_bboxes = corners.shape[0]

    coords = np.concatenate(
        [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
    )
    trans = copy.deepcopy(trans).reshape(4, 4)
    coords = coords @ trans.T

    coords = coords.reshape(-1, 4)
    # we do not filter > 0, need to keep sign of z
    if proj:
        z = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= z
        coords[:, 1] /= z
        coords[:, 2] /= np.abs(coords[:, 2])

    coords = coords[..., :3].reshape(-1, 8, 3)
    return coords


def show_box_on_views(classes, images: Tuple[Image.Image, ...],
                      boxes: LiDARInstance3DBoxes, labels, transform,
                      aug_matrix=None):
    # in `third_party/bevfusion/mmdet3d/datasets/nuscenes_dataset.py`, they use
    # (0.5, 0.5, 0) as center, however, visualize_camera assumes this center.
    bboxes_trans = box_center_shift(boxes, (0.5, 0.5, 0.5))

    vis_output = []
    for idx, img in enumerate(images):
        image = np.asarray(img)
        # the color palette for `visualize_camera` is RGB, but they draw on BGR.
        # So we make our image to BGR. This can match their color.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        trans = transform[idx]
        if aug_matrix is not None:
            trans = aug_matrix[idx] @ trans
        # mmdet3d can only save image to file.
        temp_path = tempfile.mktemp(dir=".tmp", suffix=".png")
        img_out = visualize_camera(
            temp_path, image=image, bboxes=bboxes_trans, labels=labels,
            transform=trans, classes=classes, thickness=1,
        )
        img_out = np.asarray(Image.open(temp_path))  # ensure image is loaded
        vis_output.append(Image.fromarray(img_out))
        os.remove(temp_path)
    return vis_output
