import os
import logging
import warnings
import numpy as np
from PIL import Image
from moviepy.editor import ImageSequenceClip
import concurrent.futures
logging.captureWarnings(False)
import fire


def make_video_with_pathList(
        pathList, outpath, bitrate, out_size=(1600, 900), fps=12, verbose=True):
    """Convert images from `pathList` to video, and save to `outpath`.
    """
    if len(pathList) == 0:
        return
    imgList = []
    for path in pathList:
        img = Image.open(path).convert("RGB")
        if img.size != out_size and verbose:
            warnings.warn(
                f"Your input size is {img.size}. We will resize to {out_size}.",
                RuntimeWarning)
        img = img.resize(out_size, resample=Image.BICUBIC)
        imgList.append(np.asarray(img))
    clip = ImageSequenceClip(imgList, fps=fps)
    clip.write_videofile(
        outpath, verbose=verbose, bitrate=bitrate,
        logger='bar' if verbose else None)
    clip.close()


def load_pathList(root, prefix, img_num, ext=".png"):
    paths = []
    for i in range(img_num):
        paths.append(os.path.join(root, prefix + f"_{i}" + ext))
    return paths


def process_token(token, subfix, root, outroot, view_order, img_num, bitrate, quiet):
    for sub in subfix:
        outdir = os.path.join(outroot, token + sub)
        print(f"Your video will be saved in {outdir}")
        try:
            os.makedirs(outdir)
        except FileExistsError as e:
            print(f"{outdir} exists, please assign another or delete it.")
            raise (e)
        for view in view_order:
            sub_root = os.path.join(root, f"{token}{sub}")
            prefix = f"{token}_{view}"
            outpath = os.path.join(outdir, prefix + ".mp4")
            pathList = load_pathList(sub_root, prefix, img_num)
            make_video_with_pathList(
                pathList, outpath, bitrate, verbose=not quiet)
            print(f"Your video saved to: {outpath}")  


def make_video(
    root,
    token,
    subfix="",
    bitrate="4M",
    img_num=16,
    outroot="magicdrive-t-log/submission/video",
    quiet=False,
):
    """transfer images to videos for each camera view.
    We assume the images are saved like:
    ${ROOT}/
    ├── 0f9f4a764e8649a595541ae4bf4668d63${subfix}/
    │   ├── 0f9f4a764e8649a595541ae4bf4668d63_CAM_FRONT_LEFT_0.png
    │   ├── 0f9f4a764e8649a595541ae4bf4668d63_CAM_FRONT_LEFT_1.png
    │   ├── ...
    │   ├── 0f9f4a764e8649a595541ae4bf4668d63_CAM_FRONT_LEFT_15.png
    │   ├── 0f9f4a764e8649a595541ae4bf4668d63_CAM_FRONT_0.png
    │   └── ...
    └── ...

    And videos will be saved like:
    ${OUTROOT}/
    ├── 0f9f4a764e8649a595541ae4bf4668d63${subfix}/
    │   ├── 0f9f4a764e8649a595541ae4bf4668d63_CAM_FRONT_LEFT.mp4
    │   ├── ...
    │   └── 0f9f4a764e8649a595541ae4bf4668d63_CAM_BACK_LEFT.mp4
    └── ...

    Args:
        root (str): img root, where all png images are saved there.
        token (str): sample token for the scene.
        subfix (str|list): subfix of your generation folder, e.g., _gen0, _gen1,
            _gen2, _gen3, night, rainy, sunny. Can be string or list with
            multiple strings. 
        img_num (int, optional): video length. Defaults to 16.
        outroot (str, optional): output root. Defaults to "magicdrive-t-log/video".
    """
    view_order = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
    ]
    if isinstance(subfix, str):
        subfix = [subfix]

    if token == "all":
        import mmcv
        data_infos = mmcv.load("data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_track2_eval.pkl")
        tokens = [s[0] for s in data_infos['scene_tokens']]
    else:
        tokens = [token]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
        futures = [
            executor.submit(process_token, token, subfix, root, outroot, view_order, img_num, bitrate, quiet)
            for token in tokens
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f'Generated an exception: {exc}')


if __name__ == "__main__":
    fire.Fire(make_video)
