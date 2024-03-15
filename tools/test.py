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

# fmt: off
# bypass annoying warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# fmt: on

sys.path.append(".")  # noqa
from magicdrive.runner.utils import concat_6_views, img_concat_h, img_concat_v
from magicdrive.misc.test_utils import (
    prepare_all, run_one_batch, insert_pipeline_item
)

transparent_bg = True
target_map_size = 400
# target_map_size = 800

# def output_func(x): return concat_6_views(x)


def output_func(x): return concat_6_views(x, oneline=True)
# def output_func(x): return img_concat_h(*x[:3])


def make_video_with_filenames(filenames, outname, fps=2):
    clips = [ImageClip(m).set_duration(1 / fps) for m in filenames]
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(outname, fps=fps)


@hydra.main(version_base=None, config_path="../configs",
            config_name="test_config")
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
    pipe, val_dataloader, weight_dtype = prepare_all(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.log_root, "run_config.yaml"))

    #### start ####
    total_num = 0
    batch_index = 0
    progress_bar = tqdm(
        range(len(val_dataloader) * cfg.runner.validation_times),
        desc="Steps",
    )
    os.makedirs(os.path.join(cfg.log_root, "frames"), exist_ok=True)
    for val_input in val_dataloader:
        batch_index += 1
        batch_img_index = 0
        ori_img_paths = []
        gen_img_paths = {}
        return_tuples = run_one_batch(cfg, pipe, val_input, weight_dtype,
                                      transparent_bg=transparent_bg,
                                      map_size=target_map_size)

        for map_img, ori_imgs, ori_imgs_wb, gen_imgs_list, gen_imgs_wb_list in zip(*return_tuples):
            # save map
            map_img.save(os.path.join(
                cfg.log_root, "frames",
                f"{batch_index}_{batch_img_index}_map_{total_num}.png"))

            # save ori
            if ori_imgs is not None:
                ori_img = output_func(ori_imgs)
                save_path = os.path.join(
                    cfg.log_root, "frames",
                    f"{batch_index}_{batch_img_index}_ori_{total_num}.png")
                ori_img.save(save_path)
                ori_img_paths.append(save_path)

            # save gen
            for ti, gen_imgs in enumerate(gen_imgs_list):
                gen_img = output_func(gen_imgs)
                save_path = os.path.join(
                    cfg.log_root, "frames",
                    f"{batch_index}_{batch_img_index}_gen{ti}_{total_num}.png")
                gen_img.save(save_path)
                if ti in gen_img_paths:
                    gen_img_paths[ti].append(save_path)
                else:
                    gen_img_paths[ti] = [save_path]

            if cfg.show_box:
                # save ori with box
                if ori_imgs_wb is not None:
                    ori_img_with_box = output_func(ori_imgs_wb)
                    ori_img_with_box.save(os.path.join(
                        cfg.log_root, "frames",
                        f"{batch_index}_{batch_img_index}_ori_box_{total_num}.png"))
                # save gen with box
                for ti, gen_imgs_wb in enumerate(gen_imgs_wb_list):
                    gen_img_with_box = output_func(gen_imgs_wb)
                    gen_img_with_box.save(os.path.join(
                        cfg.log_root, "frames",
                        f"{batch_index}_{batch_img_index}_gen{ti}_box_{total_num}.png"))

            total_num += 1
            batch_img_index += 1
        make_video_with_filenames(
            ori_img_paths, os.path.join(
                cfg.log_root, f"{batch_index}_{batch_img_index}_ori.mp4"),
            fps=cfg.fps)
        for k, v in gen_img_paths.items():
            make_video_with_filenames(
                v, os.path.join(
                    cfg.log_root,
                    f"{batch_index}_{batch_img_index}_gen{k}.mp4"),
                fps=cfg.fps)

        # update bar
        progress_bar.update(cfg.runner.validation_times)


if __name__ == "__main__":
    main()
