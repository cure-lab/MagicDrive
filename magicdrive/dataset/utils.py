from typing import Tuple, List
from functools import partial
import random

import torch
import numpy as np

from transformers import CLIPTokenizer
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from ..runner.utils import trans_boxes_to_views


META_KEY_LIST = [
    "gt_bboxes_3d",
    "gt_labels_3d",
    "camera_intrinsics",
    "camera2ego",
    "lidar2ego",
    "lidar2camera",
    "camera2lidar",
    "lidar2image",
    "img_aug_matrix",
    "metas",
]


def _tokenize_captions(examples, template, tokenizer=None, is_train=True):
    captions = []
    for example in examples:
        caption = template.format(**example["metas"].data)
        captions.append(caption)
    captions.append("")
    if tokenizer is None:
        return None, captions

    # pad in the collate_fn function
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="do_not_pad",
        truncation=True,
    )
    input_ids = inputs.input_ids
    # pad to the longest of current batch (might differ between cards)
    padded_tokens = tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_tensors="pt"
    ).input_ids
    return padded_tokens, captions


def ensure_canvas(coords, canvas_size: Tuple[int, int]):
    """Box with any point in range of canvas should be kept.

    Args:
        coords (_type_): _description_
        canvas_size (Tuple[int, int]): _description_

    Returns:
        np.array: mask on first axis.
    """
    (h, w) = canvas_size
    c_mask = np.any(coords[..., 2] > 0, axis=1)
    w_mask = np.any(np.logical_and(
        coords[..., 0] > 0, coords[..., 0] < w), axis=1)
    h_mask = np.any(np.logical_and(
        coords[..., 1] > 0, coords[..., 1] < h), axis=1)
    c_mask = np.logical_and(c_mask, np.logical_and(w_mask, h_mask))
    return c_mask


def ensure_positive_z(coords):
    c_mask = np.any(coords[..., 2] > 0, axis=1)
    return c_mask


def random_0_to_1(mask: np.array, num):
    assert mask.ndim == 1
    inds = np.where(mask == 0)[0].tolist()
    random.shuffle(inds)
    mask = np.copy(mask)
    mask[inds[:num]] = 1
    return mask


def _transform_all(examples, matrix_key, proj):
    """project all bbox to views, return 2d coordinates.

    Args:
        examples (List): collate_fn input.

    Returns:
        2-d list: List[List[np.array]] for B, N_cam. Can be [].
    """
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]
    # lidar2image (np.array): lidar to image view transformation
    trans_matrix = np.stack([example[matrix_key].data.numpy()
                            for example in examples], axis=0)
    # img_aug_matrix (np.array): augmentation matrix
    img_aug_matrix = np.stack([example['img_aug_matrix'].data.numpy()
                               for example in examples], axis=0)
    B, N_cam = trans_matrix.shape[:2]

    bboxes_coord = []
    # for each keyframe set
    for idx in range(B):
        # if zero, add empty list
        if len(gt_bboxes_3d[idx]) == 0:
            # keep N_cam dim for convenient
            bboxes_coord.append([None for _ in range(N_cam)])
            continue

        coords_list = trans_boxes_to_views(
            gt_bboxes_3d[idx], trans_matrix[idx], img_aug_matrix[idx], proj)
        bboxes_coord.append(coords_list)
    return bboxes_coord


def _preprocess_bbox(bbox_mode, canvas_size, examples, is_train=True,
                     view_shared=False, use_3d_filter=True, bbox_add_ratio=0,
                     bbox_add_num=0, bbox_drop_ratio=0):
    """Pre-processing for bbox
    .. code-block:: none

                                       up z
                        front x           ^
                             /            |
                            /             |
              (x1, y0, z1) + -----------  + (x1, y1, z1)
                          /|            / |
                         / |           /  |
           (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                        |  /      .   |  /
                        | / origin    | /
        left y<-------- + ----------- + (x0, y1, z0)
            (x0, y0, z0)

    Args:
        bbox_mode (str): type of bbox raw data.
            cxyz -> x1y1z1, x1y0z1, x1y1z0, x0y1z1;
            all-xyz -> all 8 corners xyz;
            owhr -> center, l, w, h, z-orientation.
        canvas_size (2-tuple): H, W of input images
        examples: collate_fn input
        view_shared: if enabled, all views share same set of bbox and output
            N_cam=1; otherwise, use projection to keep only visible bboxes.
    Return:
        in form of dict:
            bboxes (Tensor): B, N_cam, max_len, ...
            classes (LongTensor): B, N_cam, max_len
            masks: 1 for data, 0 for padding
    """
    # init data
    bboxes = []
    classes = []
    max_len = 0
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]
    gt_labels_3d: List[torch.Tensor] = [
        example["gt_labels_3d"].data for example in examples]

    # params
    B = len(gt_bboxes_3d)
    N_cam = len(examples[0]['lidar2image'].data.numpy())
    N_out = 1 if view_shared else N_cam

    bboxes_coord = None
    if not view_shared and not use_3d_filter:
        bboxes_coord = _transform_all(examples, 'lidar2image', True)
    elif not view_shared:
        bboxes_coord_3d = _transform_all(examples, 'lidar2camera', False)

    # for each keyframe set
    for idx in range(B):
        bboxes_kf = gt_bboxes_3d[idx]
        classes_kf = gt_labels_3d[idx]

        # if zero, add zero length tensor (for padding).
        if len(bboxes_kf) == 0 or (
                random.random() < bbox_drop_ratio and is_train):
            bboxes.append([None] * N_out)
            classes.append([None] * N_out)
            continue

        # whether share the boxes across views, filtered by 2d projection.
        if not view_shared:
            index_list = []  # each view has a mask
            if use_3d_filter:
                coords_list = bboxes_coord_3d[idx]
                filter_func = ensure_positive_z
            else:
                # filter bbox according to 2d projection on image canvas
                coords_list = bboxes_coord[idx]
                # judge coord by cancas_size
                filter_func = partial(ensure_canvas, canvas_size=canvas_size)
            # we do not need to handle None since we already filter for len=0
            for coords in coords_list:
                c_mask = filter_func(coords)
                if random.random() < bbox_add_ratio and is_train:
                    c_mask = random_0_to_1(c_mask, bbox_add_num)
                index_list.append(c_mask)
                max_len = max(max_len, c_mask.sum())
        else:
            # we use as mask, torch.bool is important
            index_list = [torch.ones(len(bboxes_kf), dtype=torch.bool)]
            max_len = max(max_len, len(bboxes_kf))

        # construct data
        if bbox_mode == 'cxyz':
            # x1y1z1, x1y0z1, x1y1z0, x0y1z1
            bboxes_pt = bboxes_kf.corners[:, [6, 5, 7, 2]]
        elif bbox_mode == 'all-xyz':
            bboxes_pt = bboxes_kf.corners  # n x 8 x 3
        elif bbox_mode == 'owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {bbox_mode}")
        bboxes.append([bboxes_pt[ind] for ind in index_list])
        classes.append([classes_kf[ind] for ind in index_list])
        bbox_shape = bboxes_pt.shape[1:]

    # there is no (visible) boxes in this batch
    if max_len == 0:
        return None, None

    # pad and construct mask
    # `bbox_shape` should be set correctly
    ret_bboxes = torch.zeros(B, N_out, max_len, *bbox_shape)
    # we set unknown to -1. since we have mask, it does not matter.
    ret_classes = -torch.ones(B, N_out, max_len, dtype=torch.long)
    ret_masks = torch.zeros(B, N_out, max_len, dtype=torch.bool)
    for _b in range(B):
        _bboxes = bboxes[_b]
        _classes = classes[_b]
        for _n in range(N_out):
            if _bboxes[_n] is None:
                continue  # empty for this batch
            this_box_num = len(_bboxes[_n])
            ret_bboxes[_b, _n, :this_box_num] = _bboxes[_n]
            ret_classes[_b, _n, :this_box_num] = _classes[_n]
            ret_masks[_b, _n, :this_box_num] = True

    # assemble as input format
    ret_dict = {
        "bboxes": ret_bboxes,
        "classes": ret_classes,
        "masks": ret_masks
    }
    return ret_dict, bboxes_coord


def collate_fn(
    examples: Tuple[dict, ...],
    template: str,
    tokenizer: CLIPTokenizer = None,
    is_train: bool = True,
    bbox_mode: str = None,
    bbox_view_shared: bool = False,
    bbox_drop_ratio: float = 0,
    bbox_add_ratio: float = 0,
    bbox_add_num: int = 3,
):
    """
    We need to handle:
    1. make multi-view images (img) into tensor -> [N, 6, 3, H, W]
    2. make masks (gt_masks_bev, gt_aux_bev) into tensor
        -> [N, 25 = 8 map + 10 obj + 7 aux, 200, 200]
    3. make caption (location, desctiption, timeofday) and tokenize, padding
        -> [N, pad_length]
    4. extract camera parameters (camera_intrinsics, camera2lidar)
        camera2lidar: A @ v_camera = v_lidar
        -> [N, 6, 3, 7]
    We keep other meta data as original.
    """
    if bbox_add_ratio > 0 and is_train:
        assert bbox_view_shared == False, "You cannot add any box on view shared."

    # mask
    if "gt_aux_bev" in examples[0] and examples[0]["gt_aux_bev"] is not None:
        keys = ["gt_masks_bev", "gt_aux_bev"]
        assert bbox_drop_ratio == 0, "map is not affected in bbox_drop"
    else:
        keys = ["gt_masks_bev"]
    # fmt: off
    bev_map_with_aux = torch.stack([torch.from_numpy(np.concatenate([
        example[key] for key in keys  # np array, channel-last
    ], axis=0)).float() for example in examples], dim=0)  # float32
    # fmt: on

    # camera param
    # TODO: camera2lidar should be changed to lidar2camera
    # fmt: off
    camera_param = torch.stack([torch.cat([
        example["camera_intrinsics"].data[:, :3, :3],  # 3x3 is enough
        example["camera2lidar"].data[:, :3],  # only first 3 rows meaningful
    ], dim=-1) for example in examples], dim=0)
    # fmt: on

    ret_dict = {
        "bev_map_with_aux": bev_map_with_aux,
        "camera_param": camera_param,
        "kwargs": {},
    }

    if "img" in examples[0]:
        # multi-view images
        pixel_values = torch.stack(
            [example["img"].data for example in examples])
        pixel_values = pixel_values.to(
            memory_format=torch.contiguous_format).float()
        ret_dict["pixel_values"] = pixel_values
    elif is_train:
        raise RuntimeError("For training, you should provide gt images.")

    # bboxes_3d, convert to tensor
    # here we consider:
    # 1. do we need to filter bboxes for each view? use `view_shared`
    # 2. padding for one batch of data if need (with zero), and output mask.
    # 3. what is the expected output format? dict of kwargs to bbox embedder
    canvas_size = pixel_values.shape[-2:]
    if bbox_mode is not None:
        # NOTE: both can be None
        bboxes_3d_input, bbox_view_coord = _preprocess_bbox(
            bbox_mode, canvas_size, examples, is_train=is_train,
            view_shared=bbox_view_shared, bbox_add_ratio=bbox_add_ratio,
            bbox_add_num=bbox_add_num, bbox_drop_ratio=bbox_drop_ratio)
        ret_dict["kwargs"]["bboxes_3d_data"] = bboxes_3d_input
    else:
        bbox_view_coord = None

    # captions: one real caption with one null caption
    input_ids_padded, captions = _tokenize_captions(
        examples, template, tokenizer, is_train)
    ret_dict["captions"] = captions[:-1]  # list of str
    if tokenizer is not None:
        # real captions in head; the last one is null caption
        # we omit "attention_mask": padded_tokens.attention_mask, seems useless
        ret_dict["input_ids"] = input_ids_padded[:-1]
        ret_dict["uncond_ids"] = input_ids_padded[-1:]

    # other meta data
    meta_list_dict = dict()
    for key in META_KEY_LIST:
        try:
            meta_list = [example[key] for example in examples]
            meta_list_dict[key] = meta_list
        except KeyError:
            continue
    ret_dict['meta_data'] = meta_list_dict

    return ret_dict
