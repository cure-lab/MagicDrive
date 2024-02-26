from typing import Tuple, List
import logging
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import to_pil_image, to_tensor

from diffusers import UniPCMultistepScheduler
from accelerate.tracking import GeneralTracker

from magicdrive.runner.utils import (
    visualize_map,
    img_m11_to_01,
    concat_6_views,
)
from magicdrive.misc.common import move_to
from magicdrive.misc.test_utils import draw_box_on_imgs
from magicdrive.pipeline.pipeline_bev_controlnet import (
    BEVStableDiffusionPipelineOutput,
)
from magicdrive.dataset.utils import collate_fn
from magicdrive.networks.unet_addon_rawbox import BEVControlNetModel


def format_ori_with_gen(ori_img, gen_img_list):
    formatted_images = []

    # first image is input, followed by generations.
    formatted_images.append(np.asarray(ori_img))

    for image in gen_img_list:
        formatted_images.append(np.asarray(image))

    # formatted_images = np.stack(formatted_images)
    # 0-255 np -> 0-1 tensor -> grid -> 0-255 pil -> np
    formatted_images = torchvision.utils.make_grid(
        [to_tensor(im) for im in formatted_images], nrow=1)
    formatted_images = np.asarray(
        to_pil_image(formatted_images))
    return formatted_images


class BaseValidator:
    def __init__(self, cfg, val_dataset, pipe_cls, pipe_param) -> None:
        self.cfg = cfg
        self.val_dataset = val_dataset
        self.pipe_cls = pipe_cls
        self.pipe_param = pipe_param
        logging.info(
            f"[BaseValidator] Validator use model_param: {pipe_param.keys()}")

    def validate(
        self,
        controlnet: BEVControlNetModel,
        unet,
        trackers: Tuple[GeneralTracker, ...],
        step, weight_dtype, device
    ):
        logging.info("[BaseValidator] Running validation... ")
        controlnet.eval()  # important !!!
        unet.eval()

        pipeline = self.pipe_cls.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path,
            **self.pipe_param,
            unet=unet,
            controlnet=controlnet,
            safety_checker=None,
            feature_extractor=None,  # since v1.5 has default, we need to override
            torch_dtype=weight_dtype,
        )
        # NOTE: this scheduler does not take generator as kwargs.
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline = pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)

        if self.cfg.runner.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        image_logs = []
        progress_bar = tqdm(
            range(
                0,
                len(self.cfg.runner.validation_index)
                * self.cfg.runner.validation_times,
            ),
            desc="Val Steps",
        )

        for validation_i in self.cfg.runner.validation_index:
            raw_data = self.val_dataset[validation_i]  # cannot index loader
            val_input = collate_fn(
                [raw_data], self.cfg.dataset.template, is_train=False,
                bbox_mode=self.cfg.model.bbox_mode,
                bbox_view_shared=self.cfg.model.bbox_view_shared,
            )
            # camera_emb = self._embed_camera(val_input["camera_param"])
            camera_param = val_input["camera_param"].to(weight_dtype)

            # let different prompts have the same random seed
            if self.cfg.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device).manual_seed(
                    self.cfg.seed
                )

            # for each input param, we generate several times to check variance.
            gen_list, gen_wb_list = [], []
            for _ in range(self.cfg.runner.validation_times):
                with torch.autocast("cuda"):
                    image: BEVStableDiffusionPipelineOutput = pipeline(
                        prompt=val_input["captions"],
                        image=val_input["bev_map_with_aux"],
                        camera_param=camera_param,
                        height=self.cfg.dataset.image_size[0],
                        width=self.cfg.dataset.image_size[1],
                        generator=generator,
                        bev_controlnet_kwargs=val_input["kwargs"],
                        **self.cfg.runner.pipeline_param,
                    )
                    assert len(image.images) == 1
                    image: List[Image.Image] = image.images[0]

                gen_img = concat_6_views(image)
                gen_list.append(gen_img)
                if self.cfg.runner.validation_show_box:
                    image_with_box = draw_box_on_imgs(
                        self.cfg, 0, val_input, image)
                    gen_wb_list.append(concat_6_views(image_with_box))

                progress_bar.update(1)

            # make image for 6 views and save to dict
            ori_imgs = [
                to_pil_image(img_m11_to_01(val_input["pixel_values"][0][i]))
                for i in range(6)
            ]
            ori_img = concat_6_views(ori_imgs)
            ori_img_wb = concat_6_views(
                draw_box_on_imgs(self.cfg, 0, val_input, ori_imgs))
            map_img_np = visualize_map(
                self.cfg, val_input["bev_map_with_aux"][0])
            image_logs.append(
                {
                    "map_img_np": map_img_np,  # condition
                    "gen_img_list": gen_list,  # output
                    "gen_img_wb_list": gen_wb_list,  # output
                    "ori_img": ori_img,  # input
                    "ori_img_wb": ori_img_wb,  # input
                    "validation_prompt": val_input["captions"][0],
                }
            )

        for tracker in trackers:
            if tracker.name == "tensorboard":
                for log in image_logs:
                    map_img_np = log["map_img_np"]
                    validation_prompt = log["validation_prompt"]

                    formatted_images = format_ori_with_gen(
                        log["ori_img"], log["gen_img_list"])
                    tracker.writer.add_image(
                        validation_prompt, formatted_images, step,
                        dataformats="HWC")

                    formatted_images = format_ori_with_gen(
                        log["ori_img_wb"], log["gen_img_wb_list"])
                    tracker.writer.add_image(
                        validation_prompt + "(with box)", formatted_images,
                        step, dataformats="HWC")

                    tracker.writer.add_image(
                        "map: " + validation_prompt, map_img_np, step,
                        dataformats="HWC")
            elif tracker.name == "wandb":
                raise NotImplementedError("Do not use wandb.")
                formatted_images = []

                for log in image_logs:
                    images = log["images"]
                    validation_prompt = log["validation_prompt"]
                    validation_image = log["validation_image"]

                    formatted_images.append(
                        wandb.Image(
                            validation_image,
                            caption="Controlnet conditioning"))

                    for image in images:
                        image = wandb.Image(image, caption=validation_prompt)
                        formatted_images.append(image)

                tracker.log({"validation": formatted_images})
            else:
                logging.warn(
                    f"image logging not implemented for {tracker.name}")

        return image_logs
