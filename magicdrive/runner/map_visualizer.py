from typing import Tuple, Union
import io
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import torch



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
