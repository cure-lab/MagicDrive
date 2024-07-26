"""Run generation on val set for testing.
"""

import os
import sys
import json
import copy
import hydra
from hydra.core.hydra_config import HydraConfig
import shutil
import logging
from glob import glob
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torchvision
from torchvision.transforms import InterpolationMode
from accelerate import Accelerator

sys.path.append(".")
from perception.common.ddp_utils import concat_from_everyone
from magicdrive.misc.test_utils import (
    prepare_all, run_one_batch, update_progress_bar_config,
)


def copy_save_image(tmp, cfg, gen_imgs_list, post_trans):
    tmp_all = []
    for bi, template in enumerate(tmp):
        for gen_id, gen_imgs in enumerate(gen_imgs_list[bi]):
            # for one generation with 6 views
            for idx, view in enumerate(cfg.dataset.view_order):
                # get index in label file
                filename = os.path.basename(template['filename'][idx])
                filename = Path(view) / f"_gen_{gen_id}".join(
                    os.path.splitext(filename))
                # save to path
                save_name = os.path.join(cfg.fid.img_gen_dir, filename)
                post_trans(gen_imgs[idx]).save(save_name)
            tmp_all.append(copy.deepcopy(template))
    return tmp_all


def filter_tokens(meta_list, token_set):
    to_add_tmp = []
    for meta in meta_list:
        if meta['token'] in token_set:
            continue
        else:
            to_add_tmp.append(meta)
            token_set.add(meta['token'])
    return to_add_tmp, token_set


@hydra.main(version_base=None, config_path="../../configs",
            config_name="test_config")
def main(cfg):
    if cfg.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')
    logging.info(
        f"Your config for fid:\n" + OmegaConf.to_yaml(cfg.fid, resolve=True))

    # pipeline and dataloader
    # this function also set global seed in cfg
    accelerator = Accelerator(
        mixed_precision=cfg.accelerator.mixed_precision,
        project_dir=HydraConfig.get().runtime.output_dir,
    )
    pipe, val_dataloader, weight_dtype = prepare_all(
        cfg, device=accelerator.device)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.log_root, "run_config.yaml"))
    pipe.enable_vae_slicing()
    val_dataloader = accelerator.prepare(val_dataloader)
    pipe.to(accelerator.device)

    # random states
    if cfg.runner.validation_seed_global:
        global_generator = torch.manual_seed(
            cfg.seed + accelerator.process_index)
    else:
        global_generator = None

    # prepare
    generated_token = []

    # check resume
    if os.path.exists(cfg.fid.img_gen_dir):
        raise FileExistsError(
            f"Previous results exists: {cfg.fid.img_gen_dir}."
            f"Please remove them")
    else:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            for view in cfg.dataset.view_order:
                os.makedirs(Path(cfg.fid.img_gen_dir) / view)

    # post process
    if cfg.fid.raw_output:
        post_trans = []
    else:
        post_trans = [
            torchvision.transforms.Resize(
                OmegaConf.to_container(cfg.fid.resize, resolve=True),
                interpolation=InterpolationMode.BICUBIC,
            ),
            torchvision.transforms.Pad(
                OmegaConf.to_container(cfg.fid.padding, resolve=True)
            ),
        ]
    post_trans = torchvision.transforms.Compose(post_trans)
    logging.info(f"Using post process: {post_trans}")

    # tqdm bar
    progress_bar = tqdm(
        range(len(val_dataloader)), desc="Steps", ncols=80,
        disable=not accelerator.is_main_process)
    update_progress_bar_config(
        pipe, ncols=80, disable=not accelerator.is_main_process)

    # run
    token_set = set()
    for val_input in val_dataloader:
        bs = len(val_input['meta_data']['metas'])
        accelerator.wait_for_everyone()

        # this function also set seed to as cfg
        map_img, ori_imgs, ori_imgs_wb, gen_imgs_list, \
            gen_imgs_wb_list = run_one_batch(
                cfg, pipe, val_input, weight_dtype,
                global_generator=global_generator)

        # now make labels
        tmp = []
        for bi in range(bs):
            # for one data item, we may generate several times, they
            # share label files.
            tmp.append({
                "filename": copy.deepcopy(
                    val_input['meta_data']['metas'][bi].data['filename']),
                "token": val_input['meta_data']['metas'][bi].data['token'],
            })
        # collect and save images on main process only
        if accelerator.num_processes > int(os.environ.get("LOCAL_WORLD_SIZE", accelerator.num_processes)):
            # on multi-node, we first gather data, then save on disk.
            tmp = concat_from_everyone(accelerator, tmp)
            gen_imgs_list = concat_from_everyone(accelerator, gen_imgs_list)
            if accelerator.is_main_process:
                tmp = copy_save_image(tmp, cfg, gen_imgs_list, post_trans)
            else:
                pass
        else:
            # on single-node, we save on disk, then gather label
            tmp = copy_save_image(tmp, cfg, gen_imgs_list, post_trans)
            tmp = concat_from_everyone(accelerator, tmp)
        accelerator.wait_for_everyone()

        # main process construct data.
        if accelerator.is_main_process:
            tmp, token_set = filter_tokens(tmp, token_set)
        # update bar
        progress_bar.update(1)

    # end
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    torch.hub.set_dir("../pretrained/torch_cache/")
    main()
