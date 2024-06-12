import os
import sys
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import OmegaConf
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
from PIL import ImageOps, Image
from moviepy.editor import *

import torch
import torchvision
from torchvision.transforms import InterpolationMode
from accelerate import Accelerator

# fmt: off
# bypass annoying warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# fmt: on

sys.path.append(".")  # noqa
from magicdrive.runner.utils import concat_6_views, img_concat_h, img_concat_v
from magicdrive.misc.test_utils import (
    prepare_all, run_one_batch, update_progress_bar_config,
)


def output_func(x): return concat_6_views(x, oneline=True)


def make_video_with_filenames(filenames, outname, fps=2):
    clips = [ImageClip(m).set_duration(1 / fps) for m in filenames]
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(outname, fps=fps)


@hydra.main(version_base=None, config_path="../configs",
            config_name="submit_config")
def main(cfg: DictConfig):
    if cfg.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')

    output_dir = to_absolute_path(cfg.resume_from_checkpoint)
    original_overrides = OmegaConf.load(
        os.path.join(output_dir, "hydra/overrides.yaml"))
    current_overrides = HydraConfig.get().overrides.task

    # getting the config name of this job.
    config_name = HydraConfig.get().job.config_name
    # concatenating the original overrides with the current overrides
    overrides = original_overrides + current_overrides
    # compose a new config from scratch
    cfg = hydra.compose(config_name, overrides=overrides)

    logging.info(f"Your validation index: {cfg.runner.validation_index}")

    #### setup everything ####
    accelerator = Accelerator(
        mixed_precision=cfg.accelerator.mixed_precision,
        project_dir=HydraConfig.get().runtime.output_dir,
    )
    pipe, val_dataloader, weight_dtype = prepare_all(
        cfg, device=accelerator.device)
    if accelerator.is_main_process:
        OmegaConf.save(
            config=cfg, f=os.path.join(cfg.log_root, "run_config.yaml"))
    val_dataloader = accelerator.prepare(val_dataloader)
    pipe.to(accelerator.device)

    # For submission, post transformation
    post_trans = []
    if cfg.post.resize is not None:
        post_trans.append(
            torchvision.transforms.Resize(
                OmegaConf.to_container(cfg.post.resize, resolve=True),
                interpolation=InterpolationMode.BICUBIC)
        )
    if cfg.post.padding is not None:
        post_trans.append(
            torchvision.transforms.Pad(
                OmegaConf.to_container(cfg.post.padding, resolve=True))
        )
    post_trans = torchvision.transforms.Compose(post_trans)
    logging.info(f"Using post process: {post_trans}")

    frames_root = os.path.join(cfg.log_root, "frames")
    frames_tmp_root = os.path.join(cfg.log_root, "frames_tmp")
    visualize_root = os.path.join(cfg.log_root, "video_visualization")
    if accelerator.is_main_process:
        os.makedirs(frames_root)
        os.makedirs(frames_tmp_root)
        os.makedirs(visualize_root)

    #### start ####
    batch_index = 0
    progress_bar = tqdm(
        range(len(val_dataloader) * cfg.runner.validation_times),
        desc="Steps", ncols=80, disable=not accelerator.is_main_process)
    update_progress_bar_config(
        pipe, ncols=80, disable=not accelerator.is_main_process or cfg.cloud)

    for val_input in val_dataloader:
        batch_index += 1
        gen_img_paths = {}

        accelerator.wait_for_everyone()
        return_tuples = run_one_batch(cfg, pipe, val_input, weight_dtype)

        this_token = val_input['meta_data']['metas'][0].data['token']
        frame_idx = 0
        # loop over each frame, e.g., 16
        for map_img, ori_imgs, ori_imgs_wb, gen_imgs_list, gen_imgs_wb_list in zip(*return_tuples):
            # save gen
            # loop over generation times, e.g., 4
            for ti, gen_imgs in enumerate(gen_imgs_list):
                # gen_imgs contains 6 views for "frame_idx"-th frame on "ti"-th
                # generation
                for view, gen_img in zip(cfg.dataset.view_order, gen_imgs):
                    save_path = os.path.join(
                        frames_root, f"{this_token}_gen{ti}",
                        f"{this_token}_{view}_{frame_idx}.png")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    gen_img.save(save_path)

                # start: save for visualization, can be turned off
                view_img_all = output_func(gen_imgs)
                save_path = os.path.join(
                    frames_tmp_root, f"{this_token}_gen{ti}",
                    f"{this_token}_{frame_idx}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                view_img_all.save(save_path)
                if ti in gen_img_paths:
                    gen_img_paths[ti].append(save_path)
                else:
                    gen_img_paths[ti] = [save_path]
                # end: save for visualization, can be turned off

            frame_idx += 1

        # start: for visualization, can be turned off
        os.makedirs(visualize_root, exist_ok=True)
        for k, v in gen_img_paths.items():
            make_video_with_filenames(
                v, os.path.join(
                    visualize_root,
                    f"{this_token}_gen{k}.mp4"),
                fps=cfg.fps)
        # end: for visualization, can be turned off

        # update bar
        progress_bar.update(cfg.runner.validation_times)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
