import os
import math
import logging
from omegaconf import OmegaConf
from functools import partial
from packaging import version
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from accelerate import Accelerator

from magicdrive.dataset.utils import collate_fn
from magicdrive.runner.base_validator import BaseValidator
from magicdrive.runner.utils import (
    prepare_ckpt,
    resume_all_scheduler,
    append_dims,
)
from magicdrive.misc.common import (
    move_to,
    load_module,
    deepspeed_zero_init_disabled_context_manager,
)


class BaseRunner:
    def __init__(self, cfg, accelerator, train_set, val_set) -> None:
        self.cfg = cfg
        self.accelerator: Accelerator = accelerator
        # Load models and create wrapper for stable diffusion
        # workaround for ZeRO-3, see:
        # https://github.com/huggingface/diffusers/blob/3ebbaf7c96801271f9e6c21400033b6aa5ffcf29/examples/text_to_image/train_text_to_image.py#L571
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            self._init_fixed_models(cfg)
        self._init_trainable_models(cfg)

        # set model and xformers
        self._set_model_trainable_state()
        self._set_xformer_state()
        self._set_gradient_checkpointing()

        # dataloaders
        self.train_dataset = train_set
        self.train_dataloader = None
        self.val_dataset = val_set
        self.val_dataloader = None
        self._set_dataset_loader()

        # validator
        pipe_cls = load_module(cfg.model.pipe_module)
        self.validator = BaseValidator(
            self.cfg,
            self.val_dataset,
            pipe_cls,
            pipe_param={
                "vae": self.vae,
                "text_encoder": self.text_encoder,
                "tokenizer": self.tokenizer,
            }
        )

        # param and placeholders
        self.weight_dtype = torch.float32
        self.overrode_max_train_steps = self.cfg.runner.max_train_steps is None
        self.num_update_steps_per_epoch = None  # based on train loader
        self.optimizer = None
        self.lr_scheduler = None

    def _init_fixed_models(self, cfg):
        # fmt: off
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
        # fmt: on

    def _init_trainable_models(self, cfg):
        model_cls = load_module(cfg.model.model_module)
        controlnet_param = OmegaConf.to_container(
            self.cfg.model.controlnet, resolve=True)
        self.controlnet = model_cls.from_unet(self.unet, **controlnet_param)

    def _calculate_steps(self):
        if self.train_dataloader is None:
            return  # there is no train dataloader, no need to set anything

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.cfg.runner.gradient_accumulation_steps
        )
        # here the logic takes steps as higher priority. once set, will override
        # epochs param
        if self.overrode_max_train_steps:
            self.cfg.runner.max_train_steps = (
                self.cfg.runner.num_train_epochs * self.num_update_steps_per_epoch
            )
        else:
            # make sure steps and epochs are consistent
            self.cfg.runner.num_train_epochs = math.ceil(
                self.cfg.runner.max_train_steps / self.num_update_steps_per_epoch
            )

    def _set_dataset_loader(self):
        # dataset
        collate_fn_param = {
            "tokenizer": self.tokenizer,
            "template": self.cfg.dataset.template,
            "bbox_mode": self.cfg.model.bbox_mode,
            "bbox_view_shared": self.cfg.model.bbox_view_shared,
            "bbox_drop_ratio": self.cfg.runner.bbox_drop_ratio,
            "bbox_add_ratio": self.cfg.runner.bbox_add_ratio,
            "bbox_add_num": self.cfg.runner.bbox_add_num,
        }

        if self.train_dataset is not None:
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset, shuffle=True,
                collate_fn=partial(
                    collate_fn, is_train=True, **collate_fn_param),
                batch_size=self.cfg.runner.train_batch_size,
                num_workers=self.cfg.runner.num_workers, pin_memory=True,
                prefetch_factor=self.cfg.runner.prefetch_factor,
                persistent_workers=True,
            )
        if self.val_dataset is not None:
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset, shuffle=False,
                collate_fn=partial(
                    collate_fn, is_train=False, **collate_fn_param),
                batch_size=self.cfg.runner.validation_batch_size,
                num_workers=self.cfg.runner.num_workers,
                prefetch_factor=self.cfg.runner.prefetch_factor,
            )

    def _set_model_trainable_state(self, train=True):
        # set trainable status
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controlnet.train(train)

    def _set_xformer_state(self):
        # xformer
        if self.cfg.runner.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logging.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
                self.controlnet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly")

    def _set_gradient_checkpointing(self):
        if hasattr(self.cfg.runner.enable_unet_checkpointing, "__len__"):
            self.unet.enable_gradient_checkpointing(
                self.cfg.runner.enable_unet_checkpointing)
        elif self.cfg.runner.enable_unet_checkpointing:
            self.unet.enable_gradient_checkpointing()
        if self.cfg.runner.enable_controlnet_checkpointing:
            self.controlnet.enable_gradient_checkpointing()

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
        params_to_optimize = self.controlnet.parameters()
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
        (
            self.controlnet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.controlnet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.cfg.runner.unet_in_fp16:
            self.unet.to(self.accelerator.device, dtype=self.weight_dtype)

        with torch.no_grad():
            self.accelerator.unwrap_model(self.controlnet).prepare(
                self.cfg,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder
            )

        # We need to recalculate our total training steps as the size of the
        # training dataloader may have changed.
        self._calculate_steps()

    def run(self):
        # Train!
        total_batch_size = (
            self.cfg.runner.train_batch_size
            * self.accelerator.num_processes
            * self.cfg.runner.gradient_accumulation_steps
        )

        # fmt: off
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(self.train_dataset)}")
        logging.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        logging.info(f"  Num Epochs = {self.cfg.runner.num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {self.cfg.runner.train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {self.cfg.runner.gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {self.cfg.runner.max_train_steps}")
        # fmt: on
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.cfg.resume_from_checkpoint:
            if self.cfg.resume_from_checkpoint != "latest":
                path = os.path.basename(self.cfg.resume_from_checkpoint)
            else:
                raise RuntimeError("We do not support in-place resume.")
                # Get the most recent checkpoint
                dirs = os.listdir(self.cfg.log_root)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.cfg.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                self.accelerator.print(
                    f"Resuming from checkpoint {self.cfg.resume_from_checkpoint}"
                )
                load_path = prepare_ckpt(
                    self.cfg.resume_from_checkpoint,
                    self.accelerator.is_local_main_process
                )
                self.accelerator.wait_for_everyone()  # wait
                if self.cfg.resume_reset_scheduler:
                    # reset to prevent from loading
                    self.accelerator._schedulers = []
                # load resume
                self.accelerator.load_state(load_path)
                global_step = int(path.split("-")[1])
                if self.cfg.resume_reset_scheduler:
                    # now we load some parameters for scheduler
                    resume_all_scheduler(self.lr_scheduler, load_path)
                    self.accelerator._schedulers = [self.lr_scheduler]
                initial_global_step = (
                    global_step * self.cfg.runner.gradient_accumulation_steps
                )
                first_epoch = global_step // self.num_update_steps_per_epoch
        else:
            initial_global_step = 0

        # val before train
        if self.cfg.runner.validation_before_run or self.cfg.validation_only:
            if self.accelerator.is_main_process:
                self._validation(global_step)
            self.accelerator.wait_for_everyone()
            # if validation_only, exit
            if self.cfg.validation_only:
                self.accelerator.end_training()
                return

        # start train
        progress_bar = tqdm(
            range(0, self.cfg.runner.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
            miniters=self.num_update_steps_per_epoch // self.cfg.runner.display_per_epoch,
            maxinterval=self.cfg.runner.display_per_n_min * 60,
        )
        image_logs = None
        logging.info(
            f"Starting from epoch {first_epoch} to {self.cfg.runner.num_train_epochs}")
        for epoch in range(first_epoch, self.cfg.runner.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                loss = self._train_one_stop(batch)
                if not loss.isfinite():
                    raise RuntimeError('Your loss is NaN.')
                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # validation
                    if self.accelerator.is_main_process:
                        if global_step % self.cfg.runner.validation_steps == 0:
                            self._validation(global_step)
                    self.accelerator.wait_for_everyone()
                    # save and transfer
                    if global_step % self.cfg.runner.checkpointing_steps == 0:
                        sub_dir_name = f"checkpoint-{global_step}"
                        save_path = os.path.join(
                            self.cfg.log_root, sub_dir_name
                        )
                        self.accelerator.save_state(save_path)
                        logging.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item()}
                for lri, lr in enumerate(self.lr_scheduler.get_last_lr()):
                    logs[f"lr{lri}"] = lr
                progress_bar.set_postfix(refresh=False, **logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.cfg.runner.max_train_steps:
                    break
            else:
                # on epoch end
                if self.cfg.runner.save_model_per_epoch is not None:
                    if epoch % self.cfg.runner.save_model_per_epoch == 0:
                        logging.info(
                            f"Save at step {global_step}, epoch {epoch}")
                        self.accelerator.wait_for_everyone()
                        sub_dir_name = f"weight-E{epoch}-S{global_step}"
                        self._save_model(os.path.join(
                            self.cfg.log_root, sub_dir_name
                        ))
                self.accelerator.wait_for_everyone()
                continue  # rather than break
            break  # if inner loop break, break again
        self.accelerator.wait_for_everyone()
        self._save_model()
        self.accelerator.end_training()

    def _save_model(self, root=None):
        # if self.accelerator.is_main_process:
        if root is None:
            root = self.cfg.log_root
        controlnet = self.accelerator.unwrap_model(self.controlnet)
        controlnet.save_pretrained(
            os.path.join(self.cfg.log_root, self.cfg.model.controlnet_dir))
        logging.info(f"Save your model to: {root}")

    def _add_noise(self, latents, noise, timesteps):
        if self.cfg.runner.noise_offset > 0.0:
            # noise offset in SDXL, see:
            # https://github.com/Stability-AI/generative-models/blob/45c443b316737a4ab6e40413d7794a7f5657c19f/sgm/modules/diffusionmodules/loss.py#L47
            # they dit not apply on different channels. Don't know why.
            offset = self.cfg.runner.noise_offset * append_dims(
                torch.randn(latents.shape[:2], device=latents.device),
                latents.ndim
            ).type_as(latents)
            if self.cfg.runner.train_with_same_offset:
                offset = offset[:, :1]
            noise = noise + offset
        if timesteps.ndim == 2:
            B, N = latents.shape[:2]
            bc2b = partial(rearrange, pattern="b n ... -> (b n) ...")
            b2bc = partial(rearrange, pattern="(b n) ... -> b n ...", b=B)
        elif timesteps.ndim == 1:
            def bc2b(x): return x
            def b2bc(x): return x
        noisy_latents = self.noise_scheduler.add_noise(
            bc2b(latents), bc2b(noise), bc2b(timesteps)
        )
        noisy_latents = b2bc(noisy_latents)
        return noisy_latents

    def _train_one_stop(self, batch):
        self.controlnet.train()
        with self.accelerator.accumulate(self.controlnet):
            N_cam = batch["pixel_values"].shape[1]

            # Convert images to latent space
            latents = self.vae.encode(
                rearrange(batch["pixel_values"], "b n c h w -> (b n) c h w").to(
                    dtype=self.weight_dtype
                )
            ).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = rearrange(latents, "(b n) c h w -> b n c h w", n=N_cam)

            # embed camera params, in (B, 6, 3, 7), out (B, 6, 189)
            # camera_emb = self._embed_camera(batch["camera_param"])
            camera_param = batch["camera_param"].to(self.weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            # make sure we use same noise for different views, only take the
            # first
            if self.cfg.model.train_with_same_noise:
                noise = repeat(noise[:, 0], "b ... -> b r ...", r=N_cam)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            if self.cfg.model.train_with_same_t:
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
            else:
                timesteps = torch.stack([torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ) for _ in range(N_cam)], dim=1)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self._add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
            encoder_hidden_states_uncond = self.text_encoder(
                batch
                ["uncond_ids"])[0]

            controlnet_image = batch["bev_map_with_aux"].to(
                dtype=self.weight_dtype)

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
                **batch["kwargs"],
            )
            # fmt: on

            # starting from here, we use (B n) as batch_size
            noisy_latents = rearrange(noisy_latents, "b n ... -> (b n) ...")
            if timesteps.ndim == 1:
                timesteps = repeat(timesteps, "b -> (b n)", n=N_cam)
            # Predict the noise residual
            model_pred = self.unet(
                noisy_latents,  # b x n, 4, H/8, W/8
                timesteps.reshape(-1),  # b x n
                encoder_hidden_states=encoder_hidden_states_with_cam.to(
                    dtype=self.weight_dtype
                ),  # b x n, len + 1, 768
                down_block_additional_residuals=[
                    sample.to(dtype=self.weight_dtype)
                    for sample in down_block_res_samples
                ],  # all intermedite have four dims: b x n, c, h, w
                mid_block_additional_residual=mid_block_res_sample.to(
                    dtype=self.weight_dtype
                ),  # b x n, 1280, h, w. we have 4 x 7 as mid_block_res
            ).sample

            model_pred = rearrange(model_pred, "(b n) ... -> b n ...", n=N_cam)

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
            if self.accelerator.sync_gradients:
                params_to_clip = self.controlnet.parameters()
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.cfg.runner.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(
                set_to_none=self.cfg.runner.set_grads_to_none)

        return loss

    def _validation(self, step):
        controlnet = self.accelerator.unwrap_model(self.controlnet)
        unet = self.accelerator.unwrap_model(self.unet)
        image_logs = self.validator.validate(
            controlnet, unet, self.accelerator.trackers, step,
            self.weight_dtype, self.accelerator.device)
