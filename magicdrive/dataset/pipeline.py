import os
import logging
import warnings
from typing import Any, Dict, Tuple

import h5py
import numpy as np
from numpy import random
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from PIL import Image
import PIL.ImageDraw as ImageDraw


from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.bbox import (
    CameraInstance3DBoxes,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
)

from .pipeline_utils import one_hot_decode


@PIPELINES.register_module()
class LoadBEVSegmentationM:
    '''This only loads map annotations, there is no dynamic objects.
    In this map, the origin is at lower-left corner, with x-y transposed.
                          FRONT                             RIGHT
         Nuscenes                       transposed
        --------->  LEFT   EGO   RIGHT  ----------->  BACK   EGO   FRONT
           map                            output
                    (0,0)  BACK                       (0,0)  LEFT
    Guess reason, in cv2 / PIL coord, this is a BEV as follow:
        (0,0)  LEFT

        BACK   EGO   FRONT

              RIGHT
    All masks in np channel first format.
    '''

    AUX_DATA_CH = {
        "visibility": 1,
        "center_offset": 2,
        "center_ohw": 4,
        "height": 1,
    }

    def __init__(
        self,
        dataset_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
        object_classes: Tuple[str, ...] = None,  # object_classes
        aux_data: Tuple[str, ...] = None,  # aux_data for dynamic objects
        cache_file: str = None,
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes
        self.object_classes = object_classes
        self.aux_data = aux_data
        self.lidar2canvas = np.array([
            [canvas_h / patch_h, 0, canvas_h / 2],
            [0, canvas_w / patch_w, canvas_w / 2],
            [0, 0, 1]
        ])

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)
        if cache_file and os.path.isfile(cache_file):
            logging.info(f"using data cache from: {cache_file}")
            # load to memory and ignore all possible changes.
            self.cache = cache_file
        else:
            self.cache = None
        # this should be set through main process afterwards
        self.shared_mem_cache = None

    def _get_dynamic_aux_bbox(self, aux_mask, data: Dict[str, Any]):
        '''Three aux data (7 channels in total), class-agnostic:
        1. visibility, 1-channel
        2. center-offset, 2-channel
        3. height/2, width/2, orientation, 4-channel, on bev canvas
        4. height of bbox, in lidar coordinate
        '''
        for _idx in range(len(data['gt_bboxes_3d'])):
            box = data['gt_bboxes_3d'][_idx]
            # get canvas coordinates
            # fmt:off
            _box_lidar = np.concatenate([
                box.corners[:, [0, 3, 7, 4], :2].numpy(),
                box.bottom_center[:, None, :2].numpy(),  # center
                box.corners[:, [4, 7], :2].mean(dim=1)[:, None].numpy(),  # front
                box.corners[:, [0, 4], :2].mean(dim=1)[:, None].numpy(),  # left
            ], axis=1)
            # fmt:on
            _box_canvas = np.dot(
                np.pad(_box_lidar, ((0, 0), (0, 0), (0, 1)), constant_values=1.0),
                self.lidar2canvas.T)[..., :2]  # N, 4, xy
            # in canvas coordinates
            box_canvas = _box_canvas[0, :4]
            center_canvas = _box_canvas[0, 4:5]
            front_canvas = _box_canvas[0, 5:6]
            left_canvas = _box_canvas[0, 6:7]
            # render mask
            render = Image.fromarray(np.zeros(self.canvas_size, dtype=np.uint8))
            draw = ImageDraw.Draw(render)
            draw.polygon(
                box_canvas.round().astype(np.int32).flatten().tolist(),
                fill=1)
            # construct
            tmp_mask = np.array(render) > 0
            coords = np.stack(np.meshgrid(
                np.arange(self.canvas_size[1]), np.arange(self.canvas_size[0])
            ), -1).astype(np.float32)
            _cur_ch = 0
            if "visibility" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['visibility']
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = data['visibility'][_idx]
                _cur_ch = _ch_stop
            if "center_offset" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['center_offset']
                center_offset = coords[tmp_mask] - center_canvas
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = center_offset
                _cur_ch = _ch_stop
            if "center_ohw" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['center_ohw']
                height = np.linalg.norm(front_canvas - center_canvas)
                width = np.linalg.norm(left_canvas - center_canvas)
                # yaw = box.yaw  # scaling aspect ratio, yaw does not change
                v = ((front_canvas - center_canvas) / (
                    np.linalg.norm(front_canvas - center_canvas) + 1e-6))[0]
                # yaw = - np.arctan2(v[1], v[0])  # add negative, align with mmdet coord
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = np.array([
                    height, width, v[0], v[1]])[None]
                _cur_ch = _ch_stop
            if "height" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['height']
                bbox_height = box.height.item()  # in lidar coordinate
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = np.array([
                    bbox_height])[None]
                _cur_ch = _ch_stop
        return aux_mask

    def _get_dynamic_aux(self, data: Dict[str, Any] = None) -> Any:
        '''aux data
        case 1: self.aux_data is None, return None
        case 2: data=None, set all values to zeros
        '''
        if self.aux_data is None:
            return None  # there is no aux_data

        aux_ch = sum([self.AUX_DATA_CH[aux_k] for aux_k in self.aux_data])
        if aux_ch == 0:  # there is no available aux_data
            if len(self.aux_data) != 0:
                logging.warn(f"Your aux_data: {self.aux_data} is not available")
            return None

        aux_mask = np.zeros((*self.canvas_size, aux_ch), dtype=np.float32)
        if data is not None:
            aux_mask = self._get_dynamic_aux_bbox(aux_mask, data)

        # transpose x,y and channel first format
        aux_mask = aux_mask.transpose(2, 1, 0)
        return aux_mask

    def _project_dynamic_bbox(self, dynamic_mask, data):
        '''We use PIL for projection, while CVT use cv2. The results are
        slightly different due to anti-alias of line, but should be similar.
        '''
        for cls_id, cls_name in enumerate(self.object_classes):
            # pick boxes
            cls_mask = data['gt_labels_3d'] == cls_id
            boxes = data['gt_bboxes_3d'][cls_mask]
            if len(boxes) < 1:
                continue
            # get coordinates on canvas. the order of points matters.
            bottom_corners_lidar = boxes.corners[:, [0, 3, 7, 4], :2]
            bottom_corners_canvas = np.dot(
                np.pad(bottom_corners_lidar.numpy(), ((0, 0), (0, 0), (0, 1)),
                       constant_values=1.0),
                self.lidar2canvas.T)[..., :2]  # N, 4, xy
            # draw
            render = Image.fromarray(dynamic_mask[cls_id])
            draw = ImageDraw.Draw(render)
            for box in bottom_corners_canvas:
                draw.polygon(
                    box.round().astype(np.int32).flatten().tolist(), fill=1)
            # save
            dynamic_mask[cls_id, :] = np.array(render)[:]
        return dynamic_mask

    def _project_dynamic(self, static_label, data: Dict[str, Any]) -> Any:
        """for dynamic mask, one class per channel
        case 1: data is None, set all values to zeros
        """
        # setup
        ch = len(self.object_classes)
        dynamic_mask = np.zeros((ch, *self.canvas_size), dtype=np.uint8)

        # if int, set ch=object_classes with all zeros; otherwise, project
        if data is not None:
            dynamic_mask = self._project_dynamic_bbox(dynamic_mask, data)

        # combine with static_label
        dynamic_mask = dynamic_mask.transpose(0, 2, 1)
        combined_label = np.concatenate([static_label, dynamic_mask], axis=0)
        return combined_label

    def _load_from_cache(
            self, data: Dict[str, Any], cache_dict) -> Dict[str, Any]:
        token = data['token']
        labels = one_hot_decode(
            cache_dict['gt_masks_bev_static'][token][:], len(self.classes))
        if self.object_classes is not None:
            if None in self.object_classes:
                # HACK: if None, set all values to zero
                # there is no computation, we generate on-the-fly
                final_labels = self._project_dynamic(labels, None)
                aux_labels = self._get_dynamic_aux(None)
            else:  # object_classes is list, we can get from cache_file
                final_labels = one_hot_decode(
                    cache_dict['gt_masks_bev'][token][:],
                    len(self.classes) + len(self.object_classes)
                )
                aux_labels = cache_dict['gt_aux_bev'][token][:]
            data["gt_masks_bev_static"] = labels
            data["gt_masks_bev"] = final_labels
            data["gt_aux_bev"] = aux_labels
        else:
            data["gt_masks_bev_static"] = labels
            data["gt_masks_bev"] = labels
        return data

    def _get_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        lidar2ego = data["lidar2ego"]
        ego2global = data["ego2global"]
        # lidar2global = ego2global @ lidar2ego @ point2lidar
        lidar2global = ego2global @ lidar2ego
        if "lidar_aug_matrix" in data:  # it is I if no lidar aux or no train
            lidar2point = data["lidar_aug_matrix"]
            point2lidar = np.linalg.inv(lidar2point)
            lidar2global = lidar2global @ point2lidar

        map_pose = lidar2global[:2, 3]
        patch_box = (
            map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])

        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])  # angle between v and x-axis
        patch_angle = yaw / np.pi * 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        # cut semantics from nuscenesMap
        location = data["location"]
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)  # TODO why need transpose here?
        masks = masks.astype(np.bool)

        # here we handle possible combinations of semantics
        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        if self.object_classes is not None:
            data["gt_masks_bev_static"] = labels
            final_labels = self._project_dynamic(labels, data)
            aux_labels = self._get_dynamic_aux(data)
            data["gt_masks_bev"] = final_labels
            data["gt_aux_bev"] = aux_labels
        else:
            data["gt_masks_bev_static"] = labels
            data["gt_masks_bev"] = labels
        return data

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # if set cache, use it.
        if self.shared_mem_cache:
            try:
                return self._load_from_cache(data, self.shared_mem_cache)
            except:
                pass
        if self.cache:
            try:
                with h5py.File(self.cache, 'r') as cache_file:
                    return self._load_from_cache(data, cache_file)
            except:
                pass

        # cache miss, load normally
        data = self._get_data(data)

        # if set, add this item into it.
        if self.shared_mem_cache:
            token = data['token']
            for key in self.shared_mem_cache.keys():
                self.shared_mem_cache[key][token] = data[key]
        return data


@PIPELINES.register_module()
class ObjectRangeFilterM:
    """Filter objects by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data):
        """Call function to filter objects by the range.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(
            data["gt_bboxes_3d"], (LiDARInstance3DBoxes, DepthInstance3DBoxes)
        ):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(data["gt_bboxes_3d"], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = data["gt_bboxes_3d"]
        gt_labels_3d = data["gt_labels_3d"]
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        data["gt_bboxes_3d"] = gt_bboxes_3d
        data["gt_labels_3d"] = gt_labels_3d
        if "visibility" in data:
            data["visibility"] = data["visibility"][
                mask.numpy().astype(np.bool)]

        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(point_cloud_range={self.pcd_range.tolist()})"
        return repr_str


@PIPELINES.register_module()
class ReorderMultiViewImagesM:
    """Reorder camera views.
    ori_order is from `tools/data_converter/nuscenes_converter.py`
    Args:
        order (list[str]): List of camera names.
        safe (bool, optional): if True, will check every key. Defaults to True.
    """

    ORI_ORDER = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    SAFE_KEYS = [
        "token",
        "sample_idx",
        "lidar_path",
        "sweeps",
        "timestamp",
        "location",
        "description",
        "timeofday",
        "visibility",
        "ego2global",
        "lidar2ego",
        "ann_info",
        "img_fields",
        "bbox3d_fields",
        "pts_mask_fields",
        "pts_seg_fields",
        "bbox_fields",
        "mask_fields",
        "seg_fields",
        "box_type_3d",
        "box_mode_3d",
        "img_shape",
        "ori_shape",
        "pad_shape",
        "scale_factor",
        "gt_bboxes_3d",
        "gt_labels_3d",
        "lidar_aug_matrix",
        "gt_masks_bev_static",
        "gt_masks_bev",
        "gt_aux_bev",
    ]
    REORDER_KEYS = [
        "image_paths",
        "lidar2camera",
        "lidar2image",
        "camera2ego",
        "camera_intrinsics",
        "camera2lidar",
        "filename",
        "img",
        "img_aug_matrix",
    ]
    WARN_KEYS = []

    def __init__(self, order, safe=True):
        self.order = order
        self.order_mapper = [self.ORI_ORDER.index(it) for it in self.order]
        self.safe = safe

    def reorder(self, value):
        assert len(value) == len(self.order_mapper)
        if isinstance(value, list):  # list do not support indexing by list
            return [value[i] for i in self.order_mapper]
        else:
            return value[self.order_mapper]

    def __call__(self, data):
        if self.safe:
            for k in [k for k in data.keys()]:
                if k in self.SAFE_KEYS:
                    continue
                if k in self.REORDER_KEYS:
                    data[k] = self.reorder(data[k])
                elif k in self.WARN_KEYS:
                    # it should be empty or none
                    assert not data[k], f"you need to handle {k}: {data[k]}"
                else:
                    raise RuntimeWarning(
                        f"You have unhandled key ({k}) in data")
        else:
            for k in self.REORDER_KEYS:
                if k in data:
                    data[k] = self.reorder(data[k])
        return data


@PIPELINES.register_module()
class ObjectNameFilterM:
    """Filter GT objects by their names.
    As object names in use are assigned by initialization, this only remove -1,
    i.e., unknown / unmapped classes.
    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, data):
        gt_labels_3d = data["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )
        data["gt_bboxes_3d"] = data["gt_bboxes_3d"][gt_bboxes_mask]
        data["gt_labels_3d"] = data["gt_labels_3d"][gt_bboxes_mask]
        if "visibility" in data:
            data["visibility"] = data["visibility"][gt_bboxes_mask]
        return data


@PIPELINES.register_module()
class RandomFlip3DwithViews:
    """consider ori_order from
    `bevfusion/tools/data_converter/nuscenes_converter.py`, as follows:
        ORI_ORDER = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
    We also assume images views have same size (ori & resized).
    """
    SUPPORT_TYPE = [None, 'v', 'h', 'handv', 'horv', 'hv']
    REORDER_KEYS = [
        "image_paths",
        "filename",
        "img",
        "camera_intrinsics",
        "camera2lidar",
        # "lidar2camera",
        # "lidar2image",
        # "camera2ego",
    ]
    IMG_ORI_SIZE = [1600, 900]

    VERTICLE_FLIP_ORDER = [0, 2, 1, 3, 5, 4]  # see the order above
    HORIZEONTAL_FLIP_ORDER = [3, 5, 4, 0, 2, 1]  # see the order above

    def __init__(self, flip_ratio, direction='v', reorder=True) -> None:
        """random flip bbox, bev, points, image views

        Args:
            flip_ratio (float): prob to flip. 1 means always, 0 means never.
            direction (str, optional): h (front-back) or v (left-right).
            Defaults to 'v'.
            reorder (bool, optional): whether reorder & flip camera view.
        """
        assert 0 <= flip_ratio <= 1, f"flip ratio in [0,1]. You provide {flip_ratio}"
        assert direction in self.SUPPORT_TYPE, f"direction should from {self.SUPPORT_TYPE}"
        if not reorder:
            warnings.warn(f"You should always use reorder, please check!")
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.reorder = reorder
        logging.info(
            f"[RandomFlip3DwithViews] ratio={self.flip_ratio}, "
            f"direction={self.direction}, reorder={self.reorder}")

    def _reorder_func(self, value, order):
        assert len(value) == len(order)
        if isinstance(value, list):  # list do not support indexing by list
            return [value[i] for i in order]
        else:
            return value[order]

    def reorder_data(self, data, order):
        # flip camera views
        if "img" in data:
            new_imgs = []
            for img in data['img']:
                new_imgs.append(img.transpose(Image.FLIP_LEFT_RIGHT))
            data['img'] = new_imgs
        # change ordering, left <-> right / left-front <-> right-back
        for k in self.REORDER_KEYS:
            if k in data:
                data[k] = self._reorder_func(
                    data[k], order)
        # if flip, x offset should be reversed according to image width
        if "camera_intrinsics" in data:
            params = []
            for cam_i in data['camera_intrinsics']:
                cam_i = cam_i.copy()
                cam_i[0, 2] = self.IMG_ORI_SIZE[0] - cam_i[0, 2]
                params.append(cam_i)
            data['camera_intrinsics'] = params
        return data

    def flip_vertical(self, data, rotation):
        # rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
        if "points" in data:
            data["points"].flip("vertical")
        if "gt_bboxes_3d" in data:
            data["gt_bboxes_3d"].flip("vertical")
        if "gt_masks_bev" in data:
            data["gt_masks_bev"] = data["gt_masks_bev"][:, ::-1, :].copy()
        # change camera extrinsics, camera2lidar is the axis rotation from lidar
        # to camera, we use moving axis transformations.
        if "camera2lidar" in data:
            params = []
            for c2l in data['camera2lidar']:
                c2l = np.array([  # flip x about lidar
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]) @ c2l.copy()
                # if not reorder, flipping axis ends up with left-handed
                # coordinate.
                if self.reorder:
                    c2l = c2l @ np.array([  # flip y about new axis
                        [1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]) @ np.array([  # rotz 180 degree about new axis
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ])
                params.append(c2l)
            data['camera2lidar'] = params
        if self.reorder:
            data = self.reorder_data(data, self.VERTICLE_FLIP_ORDER)
        return data, rotation

    def flip_horizontal(self, data, rotation):
        # rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
        if "points" in data:
            data["points"].flip("horizontal")
        if "gt_bboxes_3d" in data:
            data["gt_bboxes_3d"].flip("horizontal")
        if "gt_masks_bev" in data:
            data["gt_masks_bev"] = data["gt_masks_bev"][:, :, ::-1].copy()
        # change camera extrinsics, camera2lidar is the axis rotation from lidar
        # to camera, we use moving axis transformations.
        if "camera2lidar" in data:
            params = []
            for c2l in data['camera2lidar']:
                c2l = np.array([  # flip y about lidar
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]) @ c2l.copy()
                # if not reorder, flipping axis ends up with left-handed
                # coordinate.
                if self.reorder:
                    c2l = c2l @ np.array([  # flip x about new axis
                        [-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ])
                params.append(c2l)
            data['camera2lidar'] = params
        if self.reorder:
            data = self.reorder_data(data, self.HORIZEONTAL_FLIP_ORDER)
        return data, rotation

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip = random.rand() < self.flip_ratio
        if not flip or self.direction is None:
            return data

        rotation = np.eye(3)
        if self.direction == "horv":
            directions = random.choice(['h', 'v'], 1)
        elif self.direction == "handv":
            directions = ['h', 'v']
        elif self.direction == "hv":
            choices = [['h'], ['v'], ['h', 'v']]
            choice = random.choice([0, 1, 2], 1)[0]
            directions = choices[choice]
        else:
            directions = [self.direction]

        for direction in directions:
            if direction == "v":
                data, rotation = self.flip_vertical(data, rotation)
            elif direction == "h":
                data, rotation = self.flip_horizontal(data, rotation)
            else:
                raise RuntimeError(f"Unknown direction: {direction}")

        # update params depends on lidar2camera and camera_intrinsics
        if "lidar2camera" in data:
            params = []
            for c2l in data['camera2lidar']:
                c2l = c2l.copy()
                _rot = c2l[:3, :3]
                _trans = c2l[:3, 3]
                l2c = np.eye(4)
                l2c[:3, :3] = _rot.T
                l2c[:3, 3] = -_rot.T @ _trans
                params.append(l2c)
            data['lidar2camera'] = params
        if "lidar2image" in data:
            params = []
            for l2c, cam_i in zip(
                    data['lidar2camera'], data['camera_intrinsics']):
                l2c = l2c.copy()
                cam_i = cam_i.copy()
                lidar2camera_r = l2c[:3, :3]
                lidar2camera_t = l2c[:3, 3]
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = lidar2camera_t
                lidar2image = cam_i @ lidar2camera_rt.T
                params.append(lidar2image)
            data['lidar2image'] = params
        if "camera2ego" in data:
            # I don't know how to handle this, just drop.
            data.pop("camera2ego")

        data["lidar_aug_matrix"][:3, :] = rotation @ data["lidar_aug_matrix"][:3, :]
        return data
