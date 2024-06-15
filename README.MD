# MagicDrive-t

**MagicDrive video generation**. We release this version mainly for reference. Please be prepared to solve any issue. Before getting start, it is **necessary** for users to setup and understand the code in `main` branch.

## Pretrained MagicDrive-t

- pretrianed weight for [rawbox_mv2.0t_0.4.3.yaml](configs/exp/rawbox_mv2.0t_0.4.3.yaml) on [OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155157018_link_cuhk_edu_hk/Ec6fgCLRRflNo1cq7huz_vsBrJAhg4c9mdTLmsIEUMQXrg?e=iTyyf9).

## Environment Setup
> The environment should be compatible with MagicDrive (single frame). However, this codebase rely on **another version of bevfusion** (in `third_party`) and some video related python packages.

The code is tested with `Pytorch==1.10.2` and `torchvision==0.11.3`.
You should have these packages before starting. To install additional packages, follow:

```bash
cd ${ROOT}
pip install -r requirements.txt
```

We opt to install the source code for the following packages, with `cd ${FOLDER}; pip install -e .`

```bash
# install third-party
third_party/
├── bevfusion -> based on db75150
├── diffusers -> based on v0.17.1 (afcca3916)
└── xformers -> (optional) we minorly change 0.0.19 to install with pytorch1.10.2
```
If you need our xformers, please find it [here](https://github.com/cure-lab/MagicDrive/tree/main/third_party/xformers). Please read [FAQ](https://github.com/cure-lab/MagicDrive/blob/main/doc/FAQ.md) if you encounter any issues.

## Pretrained Weights

Our training are based on [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

We assume you put them at `${ROOT}/../pretrained/` as follows:

```bash
{ROOT}/../pretrained/stable-diffusion-v1-5/
├── README.md
├── feature_extractor
├── model_index.json
├── safety_checker
├── scheduler
├── text_encoder
├── tokenizer
├── unet
├── v1-5-pruned-emaonly.ckpt
├── v1-5-pruned.ckpt
├── v1-inference.yaml
└── vae
```

Pretrained weight of **MagicDrive** (image generation)

```bash
{ROOT}/../MagicDrive-pretrained/
└── SDv1.5mv-rawbox_2023-09-07_18-39_224x400
```

## Datasets

Please prepare the nuScenes dataset as [bevfusion's instructions](https://github.com/mit-han-lab/bevfusion#data-preparation). Note:
1. Run with our forked version of mmdet3d.
2. It is better to run generation ONE-BY-ONE to avoid overwrite.
3. You have to move `nuscenes_dbinfos_train.pkl` and `nuscenes_gt_database` manual from nuscenes root to `ann_file` folder like `nuscenes_mmdet3d`.

After preparation, you should have
```bash
${ROOT}/../data/
├── nuscenes
│   ├── ...
│   └── sweeps
└── nuscenes_mmdet3d
```

> [!NOTE]  
> In our latest version/model, we only adopt **Option2**. It is safe to skip **Option1**.
>
> We have released all the pre-generated annotations, please find them at [W-CODA2024 Track2](https://coda-dataset.github.io/w-coda2024/track2/#data_table).

**(Option1)** Generation `ann_file` for video frames (with keyframes / sweeps). We use them to train 7~16-frame video model.

```bash
# create `nuscenes_mmdet3d-t-keyframes`
python tools/create_data.py nuscenes \
	--root-path ../data/nuscenes --out-dir ../data/nuscenes_mmdet3d-t-keyframes/ \
	--extra-tag nuscenes --only_info

# create `nuscenes_mmdet3d-t-use-break`
USE_BREAK=True \
python tools/create_data.py nuscenes \
	--root-path ../data/nuscenes --out-dir ../data/nuscenes_mmdet3d-t-use-break/ \
	--extra-tag nuscenes --only_info --with_cam_sweeps
```

The data structure should looks like:

```bash
${ROOT}/../data/
├── ...
├── nuscenes_mmdet3d-t-use-break
│   ├── nuscenes_dbinfos_train.pkl -> ../nuscenes_mmdet3d/nuscenes_dbinfos_train.pkl
│   ├── nuscenes_gt_database -> ../nuscenes_mmdet3d/nuscenes_gt_database/
│   ├── nuscenes_infos_train_t6.pkl
│   └── nuscenes_infos_val_t6.pkl
└── nuscenes_mmdet3d-t-keyframes
    ├── nuscenes_dbinfos_train.pkl -> ../nuscenes_mmdet3d/nuscenes_dbinfos_train.pkl
    ├── nuscenes_gt_database -> ../nuscenes_mmdet3d/nuscenes_gt_database
    ├── nuscenes_infos_train.pkl
    └── nuscenes_infos_val.pkl
```

**(Option2)** Generation annotations for sweep frames and `ann_file` for MagicDrive. We will use them to train 16-frame video models, and video generation for all 13~16 frame models.

1. Please follow [ASAP](https://github.com/JeffWang987/ASAP/blob/main/docs/prepare_data.md) to generate `interp` annotations for nuScenes. Simply, the following command should do the work:
	```bash
	# in ASAP root.
	bash scripts/ann_generator.sh 12 --ann_strategy 'interp' 	
	```
2. (Optional) Generate `advanced` annotations for sweeps. (We do not observe major difference between `interp` and `advanced`. This step can be skipped.)
3. Use commands in `scripts/prepare_dataset.sh` to generate `ann_file` and cache.

You should have
```bash
${ROOT}/../data/
├── ...
├── nuscenes
│	  ├── advanced_12Hz_trainval
│	  ├── interp_12Hz_trainval
│	  ├── nuscenes_advanced_12Hz_gt_database
│	  └── nuscenes_interp_12Hz_gt_database
└── nuscenes_mmdet3d-12Hz
	  ├── nuscenes_advanced_12Hz_dbinfos_train.pkl
	  ├── nuscenes_advanced_12Hz_infos_train.pkl
	  ├── nuscenes_advanced_12Hz_infos_val.pkl
	  ├── nuscenes_interp_12Hz_dbinfos_train.pkl
	  ├── nuscenes_interp_12Hz_infos_train.pkl
	  └── nuscenes_interp_12Hz_infos_val.pkl
```

(Optional but recommended) To accelerate data loading, we prepared cache files in h5 format for BEV maps. They can be generated through `tools/prepare_map_aux.py` with config in `configs/exp/map_cache_gen.yaml`. You have to rename the cache files correctly after generating them.

```bash
${ROOT}/../data/
├── ...
├── nuscenes_map_aux  # single frame cache, keyframes also use this.
│   ├── train_26x200x200_map_aux_full.h5
│   ├── train_26x400x400_map_aux_full.h5
│   ├── val_26x200x200_map_aux_full.h5
│   └── val_26x400x400_map_aux_full.h5
├── nuscenes_map_aux_12Hz_adv  # from advanced
│		├── train_26x200x200_12Hz_advanced.h5
│ 	└── val_26x200x200_12Hz_advanced.h5
├── nuscenes_map_aux_12Hz_int  # from interp
│		├── train_26x200x200_12Hz_interp.h5
│		└── val_26x200x200_12Hz_interp.h5
└── nuscenes_map_cache_t-use-break  # with sweep, use break
		├── train_8x200x200_map_use-break.h5
		└── val_8x200x200_map_use-break.h5
```

## Train MagicDrive-t

Run training for 224x400 with 7 frames.

```bash
scripts/dist_train.sh 8 runner=8gpus_t +exp=rawbox_mv2.0t_0.3.3
```

Run training for 224x400 with 16 frames.

```bash
scripts/dist_train.sh 8 runner=8gpus_t +exp=rawbox_mv2.0t_0.3.4
```

Run training for 224x400 with 16 frames with sweeps and generated annotations.

```bash
scripts/dist_train.sh 8 runner=8gpus_t +exp=rawbox_mv2.0t_0.4.3
# or
scripts/dist_train.sh 8 runner=8gpus_t +exp=rawbox_mv2.0t_0.4.4
```

Run training for 224x400 with 61 frames with sweeps and generated annotations. (8xA800)

```bash
scripts/dist_train.sh 8 runner=8gpus_t +exp=rawbox_mv2.0t_0.4.3_60
```

**Typically, train ~80000 steps (or 4 epochs with 12Hz data) would be enough.**

## Video Generation

Our default log directory is `${ROOT}/magicdrive-t-log`. Please be prepared.

**Run video generation with 12Hz annotations.**

```bash
python tools/test.py resume_from_checkpoint=${RUN_LOG_DIR} task_id=${ANY} \
	runner.validation_times=4 runner.pipeline_param.init_noise=rand_all \
	++dataset.data.val.ann_file=${ROOT}/../data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val.pkl
```

## Cite Us

```bibtex
@inproceedings{gao2023magicdrive,
  title={{MagicDrive}: Street View Generation with Diverse 3D Geometry Control},
  author={Gao, Ruiyuan and Chen, Kai and Xie, Enze and Hong, Lanqing and Li, Zhenguo and Yeung, Dit-Yan and Xu, Qiang},
  booktitle = {International Conference on Learning Representations},
  year={2024}
}
```

## Credit

We adopt following open-sourced projects:
- [bevfusion](https://github.com/mit-han-lab/bevfusion)
- [diffusers](https://github.com/huggingface/diffusers)
- [xformers](https://github.com/facebookresearch/xformers)
