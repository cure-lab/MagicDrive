from typing import Tuple, Union
import os
import io
import cv2
import copy
import logging
import tempfile
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import torch
from accelerate.scheduler import AcceleratedScheduler

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera


# fmt: off
COLORS = {
    # static
    # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/map_expansion/map_api.py#L684
    'drivable_area':        (166, 206, 227), # #a6cee3, blue
    'drivable_area*':       (144, 196, 255), # darker blue
    'lane':                 (110, 110, 110), # grey
    'road_segment':         (90, 90, 90),    # darker grey
    'ped_crossing':         (251, 154, 153), # #fb9a99, light red
    'walkway':              (227, 26, 28),   # #e31a1c, red
    'stop_line':            (253, 191, 111), # #fdbf6f, yellow
    'carpark_area':         (255, 127, 0),   # #ff7f00, orange
    'road_block':           (178, 223, 138), # #b2df8a, green

    # dividers
    'road_divider':         (255, 200, 0),
    'lane_divider':         (130, 130, 130),

    # dynamic
    # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py
    'car':                  (255, 158, 0),  # Orange
    'truck':                (255, 99, 71),  # Tomato
    'construction_vehicle': (233, 150, 70), # Darksalmon
    'bus':                  (255, 127, 80), # Coral
    'trailer':              (255, 140, 0),  # Darkorange
    'barrier':              (112, 128, 144),# Slategrey
    'motorcycle':           (255, 61, 99),  # Red
    'bicycle':              (220, 20, 60),  # Crimson
    'pedestrian':           (0, 0, 230),    # Blue
    'traffic_cone':         (47, 79, 79),   # Darkslategrey

    'nothing':              (200, 200, 200)
}
# fmt: on


# only static layer need this, object classes do not have overlap
STATIC_PRIORITY = [
    "drivable_area",
    "drivable_area*",
    "road_block",
    "walkway",
    "stop_line",
    "carpark_area",
    "ped_crossing",
    "divider",
    "road_divider",
    "lane_divider",
]


def get_colors(semantics):
    return np.array([COLORS[s] for s in semantics], dtype=np.uint8)


def get_color_by_priority(semantics: Tuple[str, ...]):
    if len(semantics) == 0:
        return COLORS["nothing"]
    indexes = [STATIC_PRIORITY.index(semantic) for semantic in semantics]
    max_semantic = semantics[np.argmax(indexes)]
    color = get_colors([max_semantic])[0]
    return color


def rgb_to_01range(rgb: Tuple[int, int, int]):
    return [c / 255.0 for c in rgb]


def show_legend(semantic_in_use, long_edge_size=200, ncol=4):
    legendFig = plt.figure("Legend plot")
    patches = []
    for k, v in COLORS.items():
        if k in semantic_in_use:
            # matplotlib takes rgb in [0, 1] range
            patches.append(mpatches.Patch(color=rgb_to_01range(v), label=k))
    legendFig.legend(handles=patches, loc="center", ncol=ncol)
    with io.BytesIO() as img_buf:
        legendFig.savefig(img_buf, format="png", bbox_inches="tight")
        im = Image.open(img_buf)
        (w, h) = im.size
        ratio = long_edge_size / max(w, h)
        # use `long_edge_size`, make sure long size is exactly the value
        if w > h:
            resized_size = (long_edge_size, int(h * ratio))
        elif h > w:
            resized_size = (int(w * ratio), long_edge_size)
        else:
            resized_size = (long_edge_size, long_edge_size)
        im = im.resize(resized_size, resample=Image.NEAREST)
        im = np.array(im)[..., :3]  # remove alpha channel
    plt.close("all")
    return im


def render_static(static_map, static_semantic, semantic_used):
    if len(static_semantic) == 0 or None in static_semantic:
        return None, None, semantic_used
    (h, w, _) = static_map.shape
    # binary mask
    mask_static = static_map.max(-1, keepdims=True).astype(np.uint8)
    rendered_static = []
    for v in static_map.reshape(h * w, -1):  # for each position
        tmp = static_semantic[np.where(v)].tolist()  # take index by mask
        semantic_used = semantic_used.union(tmp)
        rendered_static.append(get_color_by_priority(tmp))  # assign color
    rendered_static = np.array(rendered_static).reshape(h, w, 3)
    return mask_static, rendered_static, semantic_used


def render_dynamic(dynamic_map, dynamic_semantic, semantic_used):
    if len(dynamic_semantic) == 0 or None in dynamic_semantic or dynamic_map.shape[-1] == 0:
        return None, None, semantic_used
    (h, w, _) = dynamic_map.shape
    # binary mask
    mask_dynamic = dynamic_map.max(-1, keepdims=True).astype(np.uint8)
    semantic_map = dynamic_semantic[dynamic_map.argmax(-1)]  # ignore overlap
    semantic_used = semantic_used.union(np.unique(semantic_map))
    dynamic_colors = np.array([COLORS[ds] for ds in dynamic_semantic])
    rendered_dynamic = dynamic_colors[dynamic_map.argmax(-1)]
    rendered_dynamic = rendered_dynamic.reshape(h, w, 3)
    return mask_dynamic, rendered_dynamic, semantic_used


def classes_to_np(classes):
    if classes is not None:
        semantic = np.array(classes)
    else:
        semantic = np.array([])
    return semantic


def visualize_map(
    cfg, map: Union[np.ndarray, torch.Tensor], target_size=400
) -> np.ndarray:
    """visualize bev map

    Args:
        cfg (_type_): projet cfg
        map (Union[np.ndarray, torch.Tensor]): local bev map, channel first

    Returns:
        np.ndarray: uint8 image
    """

    if isinstance(map, torch.Tensor):
        map = map.cpu().numpy()
    map = map.transpose(1, 2, 0)  # channel last

    # we assume map has static + dynamic layers, classes can be None
    static_semantic = classes_to_np(cfg.dataset.map_classes)
    dynamic_semantic = classes_to_np(cfg.dataset.object_classes)

    empty = np.uint8(COLORS["nothing"])[None, None]
    semantic_used = set()

    # static
    static_map = map[..., : len(static_semantic)]
    mask_static, rendered_static, semantic_used = render_static(
        static_map, static_semantic, semantic_used)

    # dynamic
    dynamic_map = map[
        ..., len(static_semantic): len(static_semantic) + len(dynamic_semantic)
    ]
    mask_dynamic, rendered_dynamic, semantic_used = render_dynamic(
        dynamic_map, dynamic_semantic, semantic_used)

    # combine
    if mask_dynamic is None:
        rendered = mask_static * rendered_static + (1 - mask_static) * empty
    elif mask_static is None:
        rendered = mask_dynamic * rendered_dynamic + (1 - mask_dynamic) * empty
    else:
        rendered = (
            (mask_dynamic * rendered_dynamic)
            + np.logical_and(mask_static, 1 - mask_dynamic) * rendered_static
            + (1 - np.logical_or(mask_dynamic, mask_static)) * empty
        )
    rendered = rendered.astype(np.uint8)

    # resize long edge
    rendered = Image.fromarray(rendered)
    (w, h) = rendered.size
    ratio = max(target_size / w, target_size / h)
    rendered = rendered.resize((int(w * ratio), int(h * ratio)))
    rendered = rendered.rotate(90)
    rendered = np.asarray(rendered)

    # add legend
    (h, w, _) = rendered.shape
    legend = show_legend(semantic_used, long_edge_size=target_size)
    (lh, lw, _) = legend.shape
    if lh > lw:
        final_render = np.pad(rendered, ((0, 0), (0, lw), (0, 0)))
        final_render[:, w:] = legend
    else:
        final_render = np.pad(rendered, ((0, lh), (0, 0), (0, 0)))
        final_render[h:, :] = legend

    return final_render


def concat_6_views(imgs: Tuple[Image.Image, ...], oneline=False):
    if oneline:
        image = img_concat_h(*imgs)
    else:
        image = img_concat_v(img_concat_h(*imgs[:3]), img_concat_h(*imgs[3:]))
    return image


def img_m11_to_01(img):
    return img * 0.5 + 0.5


def img_concat_h(im1, *args, color='black'):
    if len(args) == 1:
        im2 = args[0]
    else:
        im2 = img_concat_h(*args)
    height = max(im1.height, im2.height)
    mode = im1.mode
    dst = Image.new(mode, (im1.width + im2.width, height), color=color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def img_concat_v(im1, *args, color="black"):
    if len(args) == 1:
        im2 = args[0]
    else:
        im2 = img_concat_v(*args)
    width = max(im1.width, im2.width)
    mode = im1.mode
    dst = Image.new(mode, (width, im1.height + im2.height), color=color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


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


def prepare_ckpt(path, download=False):
    return path


def resume_all_scheduler(lr_scheduler: AcceleratedScheduler, ckpt_dir):
    weight = torch.load(os.path.join(ckpt_dir, "scheduler.bin"))
    keys_to_load = ["last_epoch", "_step_count", "_last_lr"]
    current_dict = lr_scheduler.state_dict()
    for key in current_dict.keys():
        if key in keys_to_load:
            current_dict[key] = weight[key]
    lr_scheduler.load_state_dict(current_dict)
    return lr_scheduler


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims
    dimensions.
    """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def smart_param_count(params) -> str:
    total_num = sum(p.numel() for p in params)
    if total_num > 1024 ** 3:
        info = f"{total_num / 1024 ** 3:.2f} G"
    elif total_num > 1024 ** 2:
        info = f"{total_num / 1024 ** 2:.2f} M"
    elif total_num >= 1024:
        info = f"{total_num / 1024:.2f} K"
    else:
        info = f"{total_num}"
    return info
