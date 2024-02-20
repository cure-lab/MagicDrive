## Demo for MagicDrive

### Data format
Please check preprocessed data in `demo/data`. Specifically, each data sample is named by its token in nuScenes and contrains:
```python
{
    'img': preprocessed image, (6, 3, 224, 400),
    'gt_bboxes_3d': bbox coordinates, (N, 9), only 0:7 are used in this project
    'gt_labels_3d': bbox labels, (N),
    'gt_masks_bev': bev map, (8, 200, 200),
    'camera_intrinsics': (6, 4, 4) for 6 cameras,
    'lidar2camera': (6, 4, 4) for 6 cameras,
    'img_aug_matrix': matrix for image preprocessing, (6, 4, 4),
    'metas': {
        'timeofday': [useless],
        'location' as in nuScenes,
        'description': as in nuScenes,
        'token': as in nuScenes
    }
}
```
for more details, please check [bevfusion](https://github.com/mit-han-lab/bevfusion).

### Run the demo
Before you run, please make sure that you have install all the dependencies and prepared the pretrained models.

Run with following command (with xformers):
```bash
python demo/run.py \
    resume_from_checkpoint=pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400
```
Alternatively, if you do not have xformers, disable it through command line:
```bash
python demo/run.py \
    resume_from_checkpoint=pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400 \
    ++runner.enable_xformers_memory_efficient_attention=false
```
The generated results will be located at `magicdrive-log/test`.

Similar to the command above, changing `run.py` to `run_cond_on_view.py` can generate camera views condition on one given view.

### Interactive GUI
Install `gradio` before running:
```bash
pip install gradio
```
Make sure you can run the demo above, then launch the GUI through:
```bash
python demo/interactive_gui.py
```