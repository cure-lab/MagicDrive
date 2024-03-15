import logging
import os
import contextlib
from functools import partial
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers.optimization import get_scheduler

from ..misc.common import load_module, convert_outputs_to_fp16, move_to
from .multiview_runner import MultiviewRunner
from .base_t_validator import BaseTValidator
from .utils import smart_param_count, prepare_ckpt
from ..networks.unet_2d_condition_multiview import UNet2DConditionModelMultiview


class MultiviewTRunner(MultiviewRunner):
    def __init__(self, cfg, accelerator, train_set, val_set) -> None:
        super().__init__(cfg, accelerator, train_set, val_set)
        pipe_cls = load_module(cfg.model.pipe_module)
        self.validator = BaseTValidator(
            self.cfg,
            self.val_dataset,
            pipe_cls,
            pipe_param={
                "vae": self.vae,
                "text_encoder": self.text_encoder,
                "tokenizer": self.tokenizer,
            }
        )
        # we set _sc_attn_index here
        if cfg.model.sc_attn_index:
            self._sc_attn_index = OmegaConf.to_container(
                cfg.model.sc_attn_index, resolve=True)
        else:
            self._sc_attn_index = None

    def get_sc_attn_index(self):
        return self._sc_attn_index

    def _init_trainable_models(self, cfg):
        unet = UNet2DConditionModelMultiview.from_pretrained(
            cfg.model.pretrained_magicdrive, subfolder=cfg.model.unet_dir)

        model_cls = load_module(cfg.model.unet_module)
        unet_param = OmegaConf.to_container(self.cfg.model.unet, resolve=True)
        self.unet = model_cls.from_unet_2d_condition(unet, **unet_param)
        if cfg.model.load_pretrain_from is not None:
            load_path = prepare_ckpt(
                cfg.model.load_pretrain_from,
                self.accelerator.is_local_main_process
            )
            self.accelerator.wait_for_everyone()  # wait
            if cfg.model.allow_partial_load:
                m, u = self.unet.load_state_dict(
                    torch.load(load_path, map_location='cpu'), strict=False)
                logging.info(
                    f"[MultiviewTRunner] weight loaded from {load_path} "
                    f"with missing: {m}, unexpected {u}.")
            else:
                self.unet.load_state_dict(
                    torch.load(load_path, map_location='cpu'))
                logging.info(
                    f"[MultiviewTRunner] weight loaded from {load_path}")

        model_cls = load_module(cfg.model.model_module)
        controlnet_param = OmegaConf.to_container(
            self.cfg.model.controlnet, resolve=True)
        self.controlnet = model_cls.from_pretrained(
            cfg.model.pretrained_magicdrive, subfolder=cfg.model.controlnet_dir,
            **controlnet_param)

        # add setter func
        for mod in self.unet.modules():
            if hasattr(mod, "_sc_attn_index"):
                mod._sc_attn_index = self.get_sc_attn_index

    def _set_model_trainable_state(self, train=True):
        # set trainable status
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controlnet.requires_grad_(False)
        self.controlnet.train(False)
        # only unet
        self.unet.requires_grad_(False)
        for name, mod in self.unet.trainable_module.items():
            logging.debug(
                f"[MultiviewRunner] set {name} to requires_grad = True")
            mod.requires_grad_(train)

    def set_optimizer_scheduler(self):
        # optimizer and lr_schedulers
        if self.cfg.runner.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        unet_params = self.unet.trainable_parameters
        param_count = smart_param_count(unet_params)
        logging.info(
            f"[MultiviewRunner] add {param_count} params from unet to optimizer.")
        params_to_optimize = list(unet_params)
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=self.cfg.runner.learning_rate,
            betas=(self.cfg.runner.adam_beta1, self.cfg.runner.adam_beta2),
            weight_decay=self.cfg.runner.adam_weight_decay,
            eps=self.cfg.runner.adam_epsilon,
        )

        # lr scheduler
        self._calculate_steps()
        # fmt: off
        self.lr_scheduler = get_scheduler(
            self.cfg.runner.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.runner.lr_warmup_steps * self.cfg.runner.gradient_accumulation_steps,
            num_training_steps=self.cfg.runner.max_train_steps * self.cfg.runner.gradient_accumulation_steps,
            num_cycles=self.cfg.runner.lr_num_cycles,
            power=self.cfg.runner.lr_power,
        )
        # fmt: on

    def prepare_device(self):
        # accelerator
        ddp_modules = (
            self.unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )
        ddp_modules = self.accelerator.prepare(*ddp_modules)
        (
            self.unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = ddp_modules

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.controlnet.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.cfg.runner.unet_in_fp16 and self.weight_dtype == torch.float16:
            self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
            # move optimized params to fp32. TODO: is this necessary?
            if self.cfg.model.use_fp32_for_unet_trainable:
                for name, mod in self.accelerator.unwrap_model(
                        self.unet).trainable_module.items():
                    logging.debug(f"[MultiviewRunner] set {name} to fp32")
                    mod.to(dtype=torch.float32)
                    mod._original_forward = mod.forward
                    # autocast intermediate is necessary since others are fp16
                    mod.forward = torch.cuda.amp.autocast(
                        dtype=torch.float16)(mod.forward)
                    # we ensure output is always fp16
                    mod.forward = convert_outputs_to_fp16(mod.forward)
            else:
                raise TypeError(
                    "There is an error/bug in accumulation wrapper, please "
                    "make all trainable param in fp32.")

        # no need for this
        # with torch.no_grad():
        #     self.accelerator.unwrap_model(self.controlnet).prepare(
        #         self.cfg,
        #         tokenizer=self.tokenizer,
        #         text_encoder=self.text_encoder
        #     )
        self.accelerator.unwrap_model(
            self.controlnet).bbox_embedder._class_tokens_set_or_warned = True

        # We need to recalculate our total training steps as the size of the
        # training dataloader may have changed.
        self._calculate_steps()

    def _save_model(self, root=None):
        if root is None:
            root = self.cfg.log_root
        # if self.accelerator.is_main_process:
        unet = self.accelerator.unwrap_model(self.unet)
        unet.save_pretrained(os.path.join(root, self.cfg.model.unet_dir))
        logging.info(f"Save your model to: {root}")

    def _train_one_step(self, batch):
        self.unet.train()
        with self.accelerator.accumulate(self.unet):
            N_frame = batch["pixel_values"].shape[1]
            N_cam = batch["pixel_values"].shape[2]

            # Convert images to latent space
            latents = self.vae.encode(
                rearrange(
                    batch["pixel_values"],
                    "b l n c h w -> (b l n) c h w").to(
                    dtype=self.weight_dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = rearrange(
                latents, "(b l n) c h w -> b l n c h w", l=N_frame, n=N_cam)

            # embed camera params, in (B, 6, 3, 7), out (B, 6, 189)
            # camera_emb = self._embed_camera(batch["camera_param"])
            camera_param = batch["camera_param"].to(self.weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            # make sure we use same noise for different views, only take the
            # first
            if self.cfg.model.train_with_same_noise:
                noise = repeat(noise[:, :, 1], "b l ... -> b l r ...", r=N_cam)
            if self.cfg.model.train_with_same_noise_t:
                noise = repeat(noise[:, 0], "b ... -> b r ...", r=N_frame)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            assert self.cfg.model.train_with_same_t
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            # add frame dim
            timesteps = repeat(timesteps, "b ... -> b r ...", r=N_frame)
            timesteps = timesteps.long()

            #### here we change (b, l, ...) to (bxl, ...) ####
            f_to_b = partial(rearrange, pattern="b l ... -> (b l) ...")
            b_to_f = partial(
                rearrange, pattern="(b l) ... -> b l ...", l=N_frame)
            latents = f_to_b(latents)
            noise = f_to_b(noise)
            timesteps = f_to_b(timesteps)
            camera_param = f_to_b(camera_param)
            if batch['kwargs']['bboxes_3d_data'] is not None:
                batch_kwargs = {
                    "bboxes_3d_data": {
                        'bboxes': f_to_b(batch['kwargs']['bboxes_3d_data']['bboxes']),
                        'classes': f_to_b(batch['kwargs']['bboxes_3d_data']['classes']),
                        'masks': f_to_b(batch['kwargs']['bboxes_3d_data']['masks']),
                    }
                }
            else:
                batch_kwargs = {"bboxes_3d_data": None}

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self._add_noise(latents, noise, timesteps)

            #### here we change (b, l, ...) to (bxl, ...) ####
            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(
                f_to_b(batch["input_ids"]))[0]
            encoder_hidden_states_uncond = self.text_encoder(
                f_to_b(batch["uncond_ids"]))[0]

            controlnet_image = batch["bev_map_with_aux"].to(
                dtype=self.weight_dtype)
            controlnet_image = f_to_b(controlnet_image)

            # fmt: off
            down_block_res_samples, mid_block_res_sample, \
            encoder_hidden_states_with_cam = self.controlnet(
                noisy_latents,  # b, N_cam, 4, H/8, W/8
                timesteps,  # b
                camera_param=camera_param,  # b, N_cam, 189
                encoder_hidden_states=encoder_hidden_states,  # b, len, 768
                encoder_hidden_states_uncond=encoder_hidden_states_uncond,  # 1, len, 768
                controlnet_cond=controlnet_image,  # b, 26, 200, 200
                return_dict=False,
                **batch_kwargs,
            )
            # fmt: on

            # starting from here, we use (B n) as batch_size
            noisy_latents = rearrange(noisy_latents, "b n ... -> (b n) ...")
            if timesteps.ndim == 1:
                timesteps = repeat(timesteps, "b -> (b n)", n=N_cam)

            # Predict the noise residual
            # NOTE: Since we fix most of the model, we cast the model to fp16 and
            # disable autocast to prevent it from falling back to fp32. Please
            # enable autocast on your customized/trainable modules.
            context = contextlib.nullcontext
            context_kwargs = {}
            if self.cfg.runner.unet_in_fp16:
                context = torch.cuda.amp.autocast
                context_kwargs = {"enabled": False}
            with context(**context_kwargs):
                model_pred = self.unet(
                    noisy_latents,  # b x n, 4, H/8, W/8
                    timesteps.reshape(-1),  # b x n
                    encoder_hidden_states=encoder_hidden_states_with_cam.to(
                        dtype=self.weight_dtype
                    ),  # b x n, len + 1, 768
                    # TODO: during training, some camera param are masked.
                    down_block_additional_residuals=[
                        sample.to(dtype=self.weight_dtype)
                        for sample in down_block_res_samples
                    ],  # all intermedite have four dims: b x n, c, h, w
                    mid_block_additional_residual=mid_block_res_sample.to(
                        dtype=self.weight_dtype
                    ),  # b x n, 1280, h, w. we have 4 x 7 as mid_block_res
                ).sample

            model_pred = rearrange(model_pred, "(b n) ... -> b n ...", n=N_cam)

            #### change dims back ####
            noise = b_to_f(noise)
            model_pred = b_to_f(model_pred)

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(
                    latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(
                model_pred.float(), target.float(), reduction='none')
            loss = loss.mean()

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients and self.cfg.runner.max_grad_norm is not None:
                params_to_clip = self.unet.parameters()
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.cfg.runner.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(
                set_to_none=self.cfg.runner.set_grads_to_none)

        return loss
