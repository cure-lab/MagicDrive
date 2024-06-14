import os
import sys
import glob
import torch
import numpy as np
import argparse
import gradio as gr
from torchvision.transforms.functional import to_pil_image
from copy import deepcopy
from PIL import Image
from hydra import compose, initialize
from functools import partial
from omegaconf import OmegaConf
from diffusers import UniPCMultistepScheduler

sys.path.append(".")
from magicdrive.misc.common import load_module
from magicdrive.runner.img_utils import concat_6_views, img_m11_to_01
from demo.helper import preprocess_fn, draw_box_on_imgs, precompute_cam_ext


def load_model_from(
        dir, weight_dtype=torch.float16, device="cuda", with_xformers=None):
    original_overrides = OmegaConf.load(
        os.path.join(dir, "hydra/overrides.yaml"))
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="test_config", overrides=original_overrides)
    pipe_param = {}

    model_cls = load_module(cfg.model.model_module)
    controlnet_path = os.path.join(dir, cfg.model.controlnet_dir)
    controlnet = model_cls.from_pretrained(
        controlnet_path, torch_dtype=weight_dtype)
    controlnet.eval()
    pipe_param["controlnet"] = controlnet

    if hasattr(cfg.model, "unet_module"):
        unet_cls = load_module(cfg.model.unet_module)
        unet_path = os.path.join(dir, cfg.model.unet_dir)
        unet = unet_cls.from_pretrained(
            unet_path, torch_dtype=weight_dtype)
        unet.eval()
        pipe_param["unet"] = unet

    pipe_cls = load_module(cfg.model.pipe_module)
    pipe = pipe_cls.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        **pipe_param,
        safety_checker=None,
        feature_extractor=None,  # since v1.5 has default, we need to override
        torch_dtype=weight_dtype
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if with_xformers is None:
        if cfg.runner.enable_xformers_memory_efficient_attention:
            pipe.enable_xformers_memory_efficient_attention()
    elif with_xformers == True:
        pipe.enable_xformers_memory_efficient_attention()
    return cfg, pipe


def run_pipe(cfg, pipe, data, seed=None, prompt="", neg_prompt=None, step=None,
             scale=None):
    assert cfg.model.bbox_mode == "all-xyz"
    assert cfg.model.bbox_view_shared == False
    collate_fn_param = {
        "tokenizer": pipe.tokenizer,
        "template": cfg.dataset.template,
    }
    preprocess = partial(preprocess_fn, **collate_fn_param)
    val_input = preprocess(data)

    if seed is None:
        seed = cfg.seed
    generator = torch.manual_seed(seed)
    weight_dtype = pipe.unet.dtype
    camera_param = val_input["camera_param"].to(weight_dtype)
    pipeline_param = {**cfg.runner.pipeline_param}
    if step is not None:
        pipeline_param["num_inference_steps"] = step
    if scale is not None:
        pipeline_param["guidance_scale"] = scale
    if prompt == "":
        prompt = val_input['captions'],
    else:
        prompt = [prompt]
    if neg_prompt is not None:
        neg_prompt = [neg_prompt]
    image = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=val_input['bev_map_with_aux'],
        camera_param=camera_param,
        height=cfg.dataset.image_size[0],
        width=cfg.dataset.image_size[1],
        generator=generator,
        bev_controlnet_kwargs=val_input['kwargs'],
        **pipeline_param
    )
    image = concat_6_views(image.images[0])
    return image


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description="MagicDrive Editing GUI.")
    parser.add_argument(
        "-m", "--model", help="pretrained model path",
        default="pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400")
    parser.add_argument(
        "-d", "--data", help="sample data pattern", default="demo/data/*.pth")
    parser.add_argument(
        "--xformers", help="enable xformers", action="store_true",
        default=False)
    args = parser.parse_args()

    data_ids = []
    for sample in glob.glob(args.data):
        data_ids.append(os.path.basename(sample).split(".")[0])

    cfg, pipe = load_model_from(args.model, with_xformers=args.xformers)
    # image = run_pipe(cfg, pipe, data, seed)
    runner = partial(run_pipe, cfg=cfg, pipe=pipe)

    editing_data = {}
    current_data = None

    def apply_editing():
        """Apply editing information to `current_data`. retrun a copy (can be
        none). 
        """
        if len(editing_data) == 0:
            return current_data
        _data = deepcopy(current_data)
        for k, v in editing_data.items():
            _data['gt_bboxes_3d'][k][:7] += np.array(v)
        return _data

    def show_annotations(with_bg=True, _data=None):
        """Show annotations with given `_data`. If none, will use current edited
        data.
        """
        if _data is None:
            _data = apply_editing()
        if _data is None:  # still none, then noting to do.
            return None
        if 'img' in _data:
            ori_imgs = [to_pil_image(img_m11_to_01(img))
                        for img in _data['img']]
        else:
            size = [cfg.dataset.image_size[1], cfg.dataset.image_size[0]]
            ori_imgs = [Image.new('RGB', size) for _ in cfg.dataset.view_order]
        box_imgs = draw_box_on_imgs(
            cfg, _data, ori_imgs, transparent_bg=not with_bg)
        box_img = concat_6_views(box_imgs)
        return box_img

    def edit_and_show(xo=0, yo=0, zo=0, lo=0, wo=0, ho=0, yawo=0, idx=None,
                      update=True, with_bg=True):
        """Record editing information and show annotation.
        """
        _edit = np.array([xo, yo, zo, lo, wo, ho, yawo], dtype=np.float32)
        if any(_edit != 0) and idx is not None:
            editing_data[idx] = _edit
        if any(_edit != 0) and idx is None:
            gr.Warning("Please select bbox before editing it.")

        if not update or current_data is None:
            return gr.Image(), gr.Image()  # keep as it is.

        data_edited = apply_editing()
        if data_edited is None:  # none, nothing to do.
            return None, None
        box_img_ori = show_annotations(with_bg, data_edited)
        box_img_sel = None
        if idx is not None:
            mask = np.ones(len(data_edited['gt_bboxes_3d']), dtype=np.bool_)
            mask[idx] = 0
            data_draw = deepcopy(data_edited)
            data_draw['gt_bboxes_3d'] = data_draw['gt_bboxes_3d'][~mask]
            data_draw['gt_labels_3d'] = data_draw['gt_labels_3d'][~mask]
            box_img_sel = show_annotations(with_bg, data_draw)
        return box_img_ori, box_img_sel

    with gr.Blocks() as demo:
        # fmt: off
        with gr.Row():
            with gr.Column():  # left column
                gr.Markdown("**Sample/Box Selection**")
                data_id = gr.Dropdown(data_ids, type="value", label="Data Selection")
                box_id = gr.Dropdown(None, type="index", label="Select BBox", interactive=True)

                gr.Markdown("**Editing Options**")
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            x_offset = gr.Slider(minimum=-30, maximum=30, value=0.0, interactive=True, label="x offset (+ -> move right in lidar coord)")
                            y_offset = gr.Slider(minimum=-30, maximum=30, value=0.0, interactive=True, label="y offset (+ -> move front in lidar coord)")
                            z_offset = gr.Slider(minimum=-5, maximum=5, value=0.0, interactive=True, label="z offset (+ -> move up in lidar coord)")
                        with gr.Column():
                            l_offset = gr.Slider(minimum=-1, maximum=1, step=0.1, value=0.0, interactive=True, label="l offset (left-right)")
                            w_offset = gr.Slider(minimum=-1, maximum=1, step=0.1, value=0.0, interactive=True, label="w offset (head-tail)")
                            h_offset = gr.Slider(minimum=-1, maximum=1, step=0.1, value=0.0, interactive=True, label="h offset (top-bottom)")
                    yaw_offset = gr.Slider(minimum=-np.pi, maximum=np.pi, value=0.0, interactive=True, label="yaw offset")

                gr.Markdown("**Generation Options**")
                with gr.Group():
                    seed = gr.Number(42, label="Seed", step=1)
                    prompt = gr.Textbox("", label="Prompt")
                    with gr.Accordion("More Options:", open=False):
                        step = gr.Number(20, label="Step", minimum=5, maximum=1000, step=1)
                        scale = gr.Number(2.0, label="Guidance Scale", minimum=0, maximum=10, step=0.5)
                        neg_prompt = gr.Textbox("", label="Negative Prompt")
                        gr.Markdown("WARNING: negative prompts are not tested! see [#42](https://github.com/cure-lab/MagicDrive/issues/42).")

            with gr.Column(): # right column
                with gr.Group():
                    box_viz = gr.Image(label="Annotation Visualization")
                    box_sel_viz = gr.Image(label="Selected Box / Generated Views")
                    with gr.Row():
                        update_on_change = gr.Checkbox(value=True, label="update on change")
                        load_gt_image = gr.Checkbox(label="load gt image")
                    with gr.Row():
                        show_anno_btn = gr.Button(value="show annotation")
                        rst_anno_btn = gr.Button(value="reset this box")
                # generated = gr.Image(label="Generated Camera Views")
                with gr.Row():
                    run_btn = gr.Button(value="Generate")
                    rst_btn = gr.Button(value="Reset")
                gr.Markdown("""
                **Usage Note:**
                1. Select your data sample. Annotations will show on the right.
                2. Select the box for editing. The selected box will show in the
                second figure on the right.
                3. Change the offsets with sliders. You should notice the change
                by enabling `update on change` or click `show annotation`
                manually.
                4. After editing, click "Generate". The generated image will
                show in the second figure on the right.
                """)
        # fmt: on

        # on load data
        @data_id.input(inputs=[data_id, load_gt_image], outputs=[
            box_id, x_offset, y_offset, z_offset, yaw_offset, box_id, box_viz,
            box_sel_viz, prompt])
        def load_data(data_id, with_bg):
            global current_data
            current_data = torch.load(f"demo/data/{data_id}.pth")
            # add `camera2lidar` and `lidar2image` from `lidar2camera`
            ori_prompt = cfg.dataset.template.format(**current_data["metas"])
            current_data = precompute_cam_ext(current_data)
            editing_data.clear()
            rendered_annotation = show_annotations(with_bg)
            return gr.Dropdown(list(range(len(current_data['gt_bboxes_3d']))),
                               type="index", label="Select BBox",
                               interactive=True), 0, 0, 0, 0, None, rendered_annotation, None, ori_prompt

        # show selected box for editing
        @box_id.input(inputs=[box_id, load_gt_image], outputs=[
            x_offset, y_offset, z_offset, l_offset, w_offset, h_offset,
            yaw_offset, box_viz, box_sel_viz])
        def select_box(box_id, with_bg):
            rendered_annotation = edit_and_show(idx=box_id, with_bg=with_bg)
            if box_id in editing_data:
                e_data = editing_data[box_id]
            else:
                e_data = [0, 0, 0, 0, 0, 0, 0]
            return *e_data, *rendered_annotation

        # reset editing information for the selected box
        @rst_anno_btn.click(inputs=[box_id, load_gt_image], outputs=[
            x_offset, y_offset, z_offset, l_offset, w_offset, h_offset,
            yaw_offset, box_viz, box_sel_viz])
        def reset_box(box_id, with_bg):
            if box_id in editing_data:
                editing_data.pop(box_id)
            rendered_annotation = edit_and_show(idx=box_id, with_bg=with_bg)
            e_data = [0, 0, 0, 0, 0, 0, 0]
            return *e_data, *rendered_annotation

        # show annotations button
        @show_anno_btn.click(inputs=[box_id, load_gt_image],
                             outputs=[box_viz, box_sel_viz])
        def show_anno_btn_click(box_id, with_bg):
            if box_id is None:
                return show_annotations(with_bg), None
            return edit_and_show(idx=box_id, with_bg=with_bg)

        # update annotations on information change
        share_kwargs = {
            "inputs": [x_offset, y_offset, z_offset, l_offset, w_offset,
                       h_offset, yaw_offset, box_id, update_on_change,
                       load_gt_image],
            "outputs": [box_viz, box_sel_viz],
        }
        x_offset.change(edit_and_show, **share_kwargs)
        y_offset.change(edit_and_show, **share_kwargs)
        z_offset.change(edit_and_show, **share_kwargs)
        l_offset.change(edit_and_show, **share_kwargs)
        w_offset.change(edit_and_show, **share_kwargs)
        h_offset.change(edit_and_show, **share_kwargs)
        yaw_offset.change(edit_and_show, **share_kwargs)

        # run the model
        @run_btn.click(inputs=[
            seed, prompt, neg_prompt, step, scale, load_gt_image],
            outputs=[box_viz, box_sel_viz])
        def run_pipe(seed_g, prompt_g, neg_prompt_g, step_g, scale_g, with_bg):
            edited_data = apply_editing()
            if edited_data is None:
                return None, None
            box_img = show_annotations(with_bg)
            generated_img = runner(
                data=edited_data, seed=seed_g, prompt=prompt_g,
                neg_prompt=neg_prompt_g, step=step_g, scale=scale_g)
            return box_img, generated_img

        # reset the whole GUI.
        @rst_btn.click(outputs=[
            data_id, box_id, x_offset, y_offset, z_offset, l_offset, w_offset,
            h_offset, yaw_offset, box_viz, box_sel_viz, prompt])
        def reset_all():
            return None, None, 0, 0, 0, 0, 0, 0, 0, None, None, None

    demo.launch(server_name="0.0.0.0", server_port=7860)
