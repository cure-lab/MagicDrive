from typing import Any, Dict
import warnings
import logging

import PIL
import numpy as np
from numpy import random

from mmdet.datasets.builder import PIPELINES


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
                new_imgs.append(img.transpose(PIL.Image.FLIP_LEFT_RIGHT))
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
