from typing import Tuple, List, Optional
from PIL import Image
from omegaconf import OmegaConf
import copy

import cv2
import torch
import numpy as np

from transformers import CLIPTokenizer

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

OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}


def rotation_3d_in_axis(points, angles, axis=0):
    """Rotate points by angles according to axis.

    Args:
        points (torch.Tensor): Points of shape (N, M, 3).
        angles (torch.Tensor): Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. Defaults to 0.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will \
            raise value error.

    Returns:
        torch.Tensor: Rotated points in shape (N, M, 3)
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = torch.stack(
            [
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = torch.stack(
            [
                torch.stack([rot_cos, -rot_sin, zeros]),
                torch.stack([rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones]),
            ]
        )
    elif axis == 0:
        rot_mat_T = torch.stack(
            [
                torch.stack([zeros, rot_cos, -rot_sin]),
                torch.stack([zeros, rot_sin, rot_cos]),
                torch.stack([ones, zeros, zeros]),
            ]
        )
    else:
        raise ValueError(f"axis should in range [0, 1, 2], got {axis}")

    return torch.einsum("aij,jka->aik", (points, rot_mat_T))


class LiDARInstance3DBoxes:
    """3D LiDAR Boxes, imported from mmdet3d.

    Note:
        The box is bottom centered, i.e. the relative position of origin in
        the box is (0.5, 0.5, 0).

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x box_dim matrix.
        box_dim (int): Number of the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw).
            Default to 7.
        with_yaw (bool): Whether the box is with yaw rotation.
            If False, the value of yaw will be set to 0 as minmax boxes.
            Default to True.
        origin (tuple[float]): The relative position of origin in the box.
            Default to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, box_dim)).to(
                dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding
            # 0 as a fake yaw and set with_yaw to False.
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()

        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def dims(self):
        """torch.Tensor: Corners of each box with size (N, 8, 3)."""
        return self.tensor[:, 3:6]

    @property
    def corners(self):
        """torch.Tensor: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

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
        """
        # TODO: rotation_3d_in_axis function do not support
        #  empty tensor currently.
        assert len(self.tensor) != 0
        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
        ).to(device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 0.5, 0]
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        corners = rotation_3d_in_axis(corners, self.tensor[:, 6], axis=2)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    def __len__(self):
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]


def visualize_camera(
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int),
                    coords[index, end].astype(np.int),
                    color or OBJECT_PALETTE[name],
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas


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
        img_out = visualize_camera(
            image=image, bboxes=bboxes_trans, labels=labels,
            transform=trans, classes=classes, thickness=1,
        )
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        vis_output.append(Image.fromarray(img_out))
    return vis_output


def draw_box_on_imgs(cfg, data, ori_imgs, transparent_bg=False) -> Tuple[Image.Image, ...]:
    if transparent_bg or ori_imgs is None:
        in_imgs = [Image.new('RGB', img.size) for img in ori_imgs]
    else:
        in_imgs = ori_imgs
    gt_bboxes_3d = data['gt_bboxes_3d']
    gt_bboxes_3d = LiDARInstance3DBoxes(
        gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0))
    out_imgs = show_box_on_views(
        OmegaConf.to_container(cfg.dataset.object_classes, resolve=True),
        in_imgs, gt_bboxes_3d, data['gt_labels_3d'].numpy(),
        data['lidar2image'].numpy(), data['img_aug_matrix'].numpy(),
    )
    if transparent_bg:
        for i in range(len(out_imgs)):
            out_imgs[i].putalpha(Image.fromarray(
                (np.any(np.asarray(out_imgs[i]) > 0, axis=2) * 255).astype(np.uint8)))
    return out_imgs


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


def ensure_positive_z(coords):
    c_mask = np.any(coords[..., 2] > 0, axis=1)
    return c_mask


def _preprocess_bbox(example):
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
    gt_bboxes_3d = example["gt_bboxes_3d"]
    gt_bboxes_3d = LiDARInstance3DBoxes(
        gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0))
    gt_labels_3d: torch.Tensor = example["gt_labels_3d"]

    # params
    N_cam = len(example['lidar2image'].numpy())
    N_out = N_cam

    # lidar2image (np.array): lidar to image view transformation
    trans_matrix = example["lidar2camera"].numpy()
    # img_aug_matrix (np.array): augmentation matrix
    img_aug_matrix = example['img_aug_matrix'].numpy()
    coords_list = trans_boxes_to_views(
        gt_bboxes_3d, trans_matrix, img_aug_matrix, False)

    # if zero, add zero length tensor (for padding).
    if len(gt_bboxes_3d) == 0:
        return None
    else:
        index_list = []
        filter_func = ensure_positive_z
        # we do not need to handle None since we already filter for len=0
        for coords in coords_list:
            c_mask = filter_func(coords)
            index_list.append(c_mask)
            max_len = max(max_len, c_mask.sum())
        # construct data
        bboxes_pt = gt_bboxes_3d.corners  # n x 8 x 3
        bboxes = [bboxes_pt[ind] for ind in index_list]
        classes = [gt_labels_3d[ind] for ind in index_list]
        bbox_shape = bboxes_pt.shape[1:]

    # there is no (visible) boxes in this batch
    if max_len == 0:
        return None

    # pad and construct mask
    # `bbox_shape` should be set correctly
    ret_bboxes = torch.zeros(N_out, max_len, *bbox_shape)
    # we set unknown to -1. since we have mask, it does not matter.
    ret_classes = -torch.ones(N_out, max_len, dtype=torch.long)
    ret_masks = torch.zeros(N_out, max_len, dtype=torch.bool)
    for _n in range(N_out):
        if bboxes[_n] is None:
            continue  # empty for this batch
        this_box_num = len(bboxes[_n])
        ret_bboxes[_n, :this_box_num] = bboxes[_n]
        ret_classes[_n, :this_box_num] = classes[_n]
        ret_masks[_n, :this_box_num] = True

    # assemble as input format
    ret_dict = {
        "bboxes": ret_bboxes.unsqueeze(0),
        "classes": ret_classes.unsqueeze(0),
        "masks": ret_masks.unsqueeze(0)
    }
    return ret_dict


def _tokenize_captions(example, template, tokenizer=None):
    caption = template.format(**example["metas"])
    captions = [caption, ""]
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


def precompute_cam_ext(example):
    # fmt: off
    camera2lidar = torch.eye(4, dtype=example['lidar2camera'].dtype)
    camera2lidar = torch.stack([camera2lidar] * len(example['lidar2camera']))
    camera2lidar[:, :3, :3] = example['lidar2camera'][:, :3, :3].transpose(1, 2)
    camera2lidar[:, :3, 3:] = torch.bmm(-camera2lidar[:, :3, :3], example['lidar2camera'][:, :3, 3:])
    example['camera2lidar'] = camera2lidar
    example['lidar2image'] = torch.bmm(example['camera_intrinsics'], example['lidar2camera'])
    # fmt: on
    return example


def preprocess_fn(
    example: dict,
    template: str,
    tokenizer: CLIPTokenizer = None,
):
    """
    We need to handle:
    1. make multi-view images (img) into tensor -> [N, 6, 3, H, W]
    2. make masks (gt_masks_bev, gt_aux_bev) into tensor
        -> [N, 25 = 8 map + 10 obj + 7 aux, 200, 200]
    3. extract camera parameters (camera_intrinsics, camera2lidar)
        camera2lidar: A @ v_camera = v_lidar
        -> [N, 6, 3, 7]
    4. make caption (location, desctiption, timeofday) and tokenize, padding
        -> [N, pad_length]
    5. convert 3d bbox (filter by views, padding to same length...)
    We keep other meta data as original.
    """

    # multi-view images
    if "img" in example:
        # multi-view images
        pixel_values = example["img"]
        pixel_values = pixel_values.to(
            memory_format=torch.contiguous_format).float()
        pixel_values = pixel_values.unsqueeze(0)
    else:
        pixel_values = None

    # mask
    # np array, channel-last -> tensor
    bev_map_with_aux = torch.from_numpy(
        example["gt_masks_bev"]).float().unsqueeze(0)  # float32

    # camera param
    # TODO: camera2lidar should be changed to lidar2camera
    # fmt: off
    camera_param = torch.cat([
        example["camera_intrinsics"][:, :3, :3],  # 3x3 is enough
        example["camera2lidar"][:, :3],  # only first 3 rows meaningful
    ], dim=-1).unsqueeze(0) 
    # fmt: on

    ret_dict = {
        "pixel_values": pixel_values,
        "bev_map_with_aux": bev_map_with_aux,
        "camera_param": camera_param,
        "kwargs": {},
    }

    # captions: one real caption with one null caption
    input_ids_padded, captions = _tokenize_captions(
        example, template, tokenizer)
    ret_dict["captions"] = captions[:-1]  # list of str
    if tokenizer is not None:
        # real captions in head; the last one is null caption
        # we omit "attention_mask": padded_tokens.attention_mask, seems useless
        ret_dict["input_ids"] = input_ids_padded[:-1]
        ret_dict["uncond_ids"] = input_ids_padded[-1:]

    # bboxes_3d, convert to tensor
    # here we consider:
    # 1. do we need to filter bboxes for each view? use `view_shared`
    # 2. padding for one batch of data if need (with zero), and output mask.
    # 3. what is the expected output format? dict of kwargs to bbox embedder
    # NOTE: both can be None
    bboxes_3d_input = _preprocess_bbox(example)
    ret_dict["kwargs"]["bboxes_3d_data"] = bboxes_3d_input

    # other meta data
    meta_list_dict = dict()
    for key in META_KEY_LIST:
        try:
            meta_list = [example[key]]
            meta_list_dict[key] = meta_list
        except KeyError:
            continue
    ret_dict['meta_data'] = meta_list_dict

    return ret_dict
