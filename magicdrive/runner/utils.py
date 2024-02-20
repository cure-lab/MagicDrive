from typing import Tuple
import os
import logging
from PIL import Image

import torch
from accelerate.scheduler import AcceleratedScheduler

from magicdrive.runner.map_visualizer import *
from magicdrive.runner.box_visualizer import *
from magicdrive.runner.img_utils import *


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
