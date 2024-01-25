from typing import Tuple, Union, List
import os
import sys
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import OmegaConf
from omegaconf import DictConfig
from tqdm import tqdm
from PIL import Image

import torch
from einops import rearrange

# fmt: off
# bypass annoying warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# fmt: on

sys.path.append(".")  # noqa
from magicdrive.runner.utils import concat_6_views
from magicdrive.misc.test_utils import (
    prepare_all, run_one_batch
)
from magicdrive.pipeline import (
    StableDiffusionBEVControlNetGivenViewPipeline,
    BEVStableDiffusionPipelineOutput,
)


def run_one_batch_pipe_given_view(
    cfg,
    pipe: StableDiffusionBEVControlNetGivenViewPipeline,
    pixel_values: torch.FloatTensor,  # useless
    captions: Union[str, List[str]],
    bev_map_with_aux: torch.FloatTensor,
    camera_param: Union[torch.Tensor, None],
    bev_controlnet_kwargs: dict,
    global_generator=None
):
    """call pipe several times to generate images

    Args:
        cfg (_type_): _description_
        pipe (StableDiffusionBEVControlNetPipeline): _description_
        captions (Union[str, List[str]]): _description_
        bev_map_with_aux (torch.FloatTensor): (B=1, C=26, 200, 200), float32
        camera_param (Union[torch.Tensor, None]): (B=1, N=6, 3, 7), if None, 
            use learned embedding for uncond_cam

    Returns:
        List[List[List[Image.Image]]]: 3-dim list of PIL Image: B, Times, views
    """
    if cfg.fix_seed_for_every_generation:
        assert global_generator is None

    # let different prompts have the same random seed
    if cfg.seed is None:
        generator = None
    else:
        if global_generator is not None:
            local_seed = torch.randint(
                0x7ffffffffffffff0, [1], generator=global_generator).item()
            logging.debug(f"Using seed: {local_seed}")
            generator = torch.manual_seed(local_seed)
        else:
            generator = torch.manual_seed(cfg.seed)

    # for each input param, we generate several times to check variance.
    if isinstance(captions, str):
        batch_size = 1
    else:
        batch_size = len(captions)

    N_cam = pixel_values.shape[1]
    with torch.no_grad():
        latents = pipe.vae.encode(
            rearrange(pixel_values, "b n c h w -> (b n) c h w").to(
                dtype=pipe.vae.dtype, device=pipe._execution_device
            )
        ).latent_dist.mean
        latents = latents * pipe.vae.config.scaling_factor
        latents = rearrange(latents, "(b n) c h w -> b n c h w", n=N_cam)

    gen_imgs_list = [[] for _ in range(batch_size)]
    # last one without condition.
    for ti in range(cfg.runner.validation_times - 1):
        # reset all to none
        conditional_latents = [[None,] * N_cam for _ in range(batch_size)]
        for b in range(batch_size):
            conditional_latents[b][ti] = latents[b, ti]

        # we move here to have same seed for different generation times
        if cfg.seed is not None and cfg.fix_seed_for_every_generation:
            generator = torch.manual_seed(cfg.seed)

        image: BEVStableDiffusionPipelineOutput = pipe(
            prompt=captions,
            image=bev_map_with_aux,
            camera_param=camera_param,
            height=cfg.dataset.image_size[0],
            width=cfg.dataset.image_size[1],
            conditional_latents=conditional_latents,
            generator=generator,
            bev_controlnet_kwargs=bev_controlnet_kwargs,
            **cfg.runner.pipeline_param,
        )
        image: List[List[Image.Image]] = image.images
        for bi, imgs in enumerate(image):
            gen_imgs_list[bi].append(imgs)
    return gen_imgs_list


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
    # auto reassign pipe_module, should check.
    assert cfg.model.pipe_module == "magicdrive.pipeline.pipeline_bev_controlnet.StableDiffusionBEVControlNetPipeline"
    cfg.model.pipe_module = "magicdrive.pipeline.pipeline_bev_controlnet_given_view.StableDiffusionBEVControlNetGivenViewPipeline"
    cfg.runner.validation_index = "demo"

    #### setup everything ####
    pipe, val_dataloader, weight_dtype = prepare_all(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.log_root, "run_config.yaml"))

    #### start ####
    total_num = 0
    progress_bar = tqdm(
        range(len(val_dataloader) * cfg.runner.validation_times),
        desc="Steps",
    )
    for val_input in val_dataloader:
        return_tuples = run_one_batch(
            cfg, pipe, val_input, weight_dtype,
            run_one_batch_pipe_func=run_one_batch_pipe_given_view,
            transparent_bg=True)

        for map_img, ori_imgs, ori_imgs_wb, gen_imgs_list, gen_imgs_wb_list in zip(*return_tuples):
            # save map
            map_img.save(os.path.join(cfg.log_root, f"{total_num}_map.png"))

            # save ori
            ori_img = concat_6_views(ori_imgs, oneline=True)
            ori_img.save(os.path.join(cfg.log_root, f"{total_num}_ori.png"))
            # save gen
            for ti, gen_imgs in enumerate(gen_imgs_list):
                gen_img = concat_6_views(gen_imgs, oneline=True)
                gen_img.save(os.path.join(
                    cfg.log_root, f"{total_num}_gen{ti}.png"))
            if cfg.show_box:
                # save ori with box
                ori_img_with_box = concat_6_views(ori_imgs_wb, oneline=True)
                ori_img_with_box.save(os.path.join(
                    cfg.log_root, f"{total_num}_ori_box.png"))
                # save gen with box
                for ti, gen_imgs_wb in enumerate(gen_imgs_wb_list):
                    gen_img_with_box = concat_6_views(gen_imgs_wb, oneline=True)
                    gen_img_with_box.save(os.path.join(
                        cfg.log_root, f"{total_num}_gen{ti}_box.png"))

            total_num += 1

        # update bar
        progress_bar.update(cfg.runner.validation_times)


if __name__ == "__main__":
    main()
