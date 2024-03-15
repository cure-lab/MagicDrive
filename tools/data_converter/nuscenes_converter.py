import os
import time
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset

nus_categories = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)

nus_attributes = (
    "cycle.with_rider",
    "cycle.without_rider",
    "pedestrian.moving",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
    "vehicle.moving",
    "vehicle.parked",
    "vehicle.stopped",
    "None",
)

camera_types = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

frames_t = 6
use_break = eval(os.environ.get("USE_BREAK", "False"))


def create_nuscenes_infos(
    root_path, info_prefix, version="v1.0-trainval", max_sweeps=10,
    out_path=None, with_cam_sweeps=False,
):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    out_path = out_path if out_path else root_path

    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini",
                      "interp_12Hz_trainval", "advanced_12Hz_trainval"]
    assert version in available_vers
    if version in ["v1.0-trainval", "interp_12Hz_trainval",
                   "advanced_12Hz_trainval"]:
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"]
         for s in val_scenes])

    test = "test" in version
    if test:
        print("test scene: {}".format(len(train_scenes)))
    else:
        print("train scene: {}, val scene: {}".format(
            len(train_scenes), len(val_scenes)))
    train_nusc_infos, train_scene_tokens, val_nusc_infos, val_scene_tokens = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps,
        with_cam_sweeps=with_cam_sweeps)

    metadata = dict(version=version)
    if test:
        print("test sample: {}".format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path, "{}_infos_test.pkl".format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print(
            "train sample: {}, val sample: {}".format(
                len(train_nusc_infos), len(val_nusc_infos)
            )
        )
        data = dict(
            infos=train_nusc_infos, metadata=metadata,
            scene_tokens=train_scene_tokens)
        if with_cam_sweeps:
            filename = f"{info_prefix}_infos_train_t{frames_t}.pkl"
        else:
            filename = f"{info_prefix}_infos_train.pkl"
        info_path = osp.join(out_path, filename)
        mmcv.dump(data, info_path)
        data["infos"] = val_nusc_infos
        data["scene_tokens"] = val_scene_tokens
        if with_cam_sweeps:
            filename = f"{info_prefix}_infos_val_t{frames_t}.pkl"
        else:
            filename = f"{info_prefix}_infos_val.pkl"
        info_val_path = osp.join(out_path, filename)
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print("total scene num: {}".format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f"{os.getcwd()}/")[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            print(f"scene {scene} not exists.")
            continue
        available_scenes.append(scene)
    print("exist scene num: {}".format(len(available_scenes)))
    return available_scenes


def isTimeFormat(data: str, format: str) -> bool:
    try:
        time.strptime(data, format)
        return True
    except ValueError:
        return False


def _sample_data_transform(
        nusc, sample, max_sweeps, ego_pose_token=None, cam_tokens=None,
        keyframe=True):
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    if ego_pose_token is None:
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    else:
        pose_record = nusc.get("ego_pose", ego_pose_token)

    location = nusc.get(
        "log", nusc.get("scene", sample["scene_token"])["log_token"]
    )["location"]
    description = nusc.get("scene", sample["scene_token"])["description"]
    timeofday = nusc.get(
        "log", nusc.get("scene", sample["scene_token"])["log_token"]
    )["logfile"][5:]
    assert isTimeFormat(timeofday, '%Y-%m-%d-%H-%M-%S%z')

    if keyframe:
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        mmcv.check_file_exist(lidar_path)
    else:
        lidar_path = None
        boxes = []

    info = {
        "lidar_path": lidar_path,
        "token": sample["token"],
        "sweeps": [],
        "cams": dict(),
        "lidar2ego_translation": cs_record["translation"],
        "lidar2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sample["timestamp"],
        "location": location,
        "description": description,
        "timeofday": timeofday,
        "is_key_frame": keyframe,
    }

    l2e_r = info["lidar2ego_rotation"]
    l2e_t = info["lidar2ego_translation"]
    e2g_r = info["ego2global_rotation"]
    e2g_t = info["ego2global_translation"]
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # obtain 6 image's information per frame
    cam_dict = {}
    for cami in range(len(camera_types)):
        cam = camera_types[cami]
        if cam_tokens is None:
            cam_token = sample["data"][cam]
        else:
            cam_token = cam_tokens[cami]
        cam_dict[cam] = cam_token
    for cam, cam_token in cam_dict.items():
        cam_path, _, camera_intrinsics = nusc.get_sample_data(cam_token)
        cam_info = obtain_sensor2top(
            nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
        )
        cam_info.update(camera_intrinsics=camera_intrinsics)
        info["cams"].update({cam: cam_info})

    # obtain sweeps for a single key-frame
    sweeps = []
    sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    while len(sweeps) < max_sweeps:
        if not sd_rec["prev"] == "":
            sweep = obtain_sensor2top(
                nusc, sd_rec["prev"],
                l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, "lidar")
            sweeps.append(sweep)
            sd_rec = nusc.get("sample_data", sd_rec["prev"])
        else:
            break
    info["sweeps"] = sweeps
    # obtain annotation
    if keyframe:
        annotations = [
            nusc.get("sample_annotation", token) for token in sample["anns"]
        ]
        velocity = []
        for token in sample["anns"]:
            vel = np.array([np.nan, np.nan, np.nan])[:2]
            try:
                vel = nusc.box_velocity(token)[:2]
            except KeyError:
                print(f"No velocity on {token} in sample: {sample['token']}")
            velocity.append(vel)
        velocity = np.array(velocity)
        valid_flag = np.array(
            [
                (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                for anno in annotations
            ],
            dtype=bool,
        ).reshape(-1)
    else:
        annotations = []
        velocity = np.array([])
        valid_flag = np.array([], dtype=bool).reshape(-1)

    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(
        -1, 1
    )

    # convert velo from global to lidar
    for i in range(len(boxes)):
        velo = np.array([*velocity[i], 0.0])
        velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        velocity[i] = velo[:2]

    names = [b.name for b in boxes]
    for i in range(len(names)):
        if names[i] in NuScenesDataset.NameMapping:
            names[i] = NuScenesDataset.NameMapping[names[i]]
    names = np.array(names)
    # we need to convert rot to SECOND format.
    gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
    assert len(gt_boxes) == len(
        annotations
    ), f"{len(gt_boxes)}, {len(annotations)}"
    info["gt_boxes"] = gt_boxes
    info["gt_names"] = names
    info["gt_velocity"] = velocity.reshape(-1, 2)
    info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
    info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
    info["visibility"] = np.array(
        [a["visibility_token"] for a in annotations],
        dtype=np.uint8)
    info["valid_flag"] = valid_flag
    return info


def _get_pn_cams(nusc, direction, cams=None, sample=None):
    if cams is None and sample is None:
        raise ValueError()
    if sample is not None:
        assert cams is None
        cams = []
        for cam in camera_types:
            cams.append(nusc.get("sample_data", sample['data'][cam]))
    _cams = []
    for cam in cams:
        if cam[direction]:
            _cams.append(nusc.get("sample_data", cam[direction]))
        else:
            return None
    return _cams


def _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test=False, max_sweeps=10,
        with_cam_sweeps=True):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    train_scene_tokens = []
    val_nusc_infos = []
    val_scene_tokens = []
    scene_token = None
    _scene_tokens = []
    mid_head = None
    breaks = 0
    prev_sample = None

    if not with_cam_sweeps:
        assert use_break == False
    if use_break:
        print("generation with break in scene")
    else:
        print("generation without break in scene")

    for sample in mmcv.track_iter_progress(nusc.sample):
        # check if new sample is continuous with previous one
        if sample['scene_token'] != scene_token or mid_head is not None:
            # if not, we add and reset `_scene_tokens`
            if len(_scene_tokens) > frames_t:
                if scene_token in train_scenes:
                    train_scene_tokens.append(_scene_tokens)
                else:
                    val_scene_tokens.append(_scene_tokens)
            if sample['scene_token'] == scene_token:
                _scene_tokens = [mid_head]  # mid_head must not be None
            else:
                _scene_tokens = []
            mid_head = None
            scene_token = sample['scene_token']
        elif prev_sample is not None:
            assert prev_sample['next'] == sample['token']

        # start to get data
        infos = []
        info = _sample_data_transform(
            nusc, sample, max_sweeps=max_sweeps, keyframe=True)
        assert ";" not in info['token']
        infos.append(info)

        if with_cam_sweeps:
            # get data for previous frames
            cams = _get_pn_cams(nusc, direction='prev', sample=sample)
            frame_cnt = 1
            while cams is not None and cams[0]['sample_token'] == sample['token']:
                cam_tokens = [cam['token'] for cam in cams]
                info = _sample_data_transform(
                    nusc, sample, max_sweeps=-1,
                    ego_pose_token=cams[0]["ego_pose_token"],
                    cam_tokens=cam_tokens, keyframe=False)
                # NOTE: there is no token for non-keyframe cam scenes. while ";"
                # never appears in token, we add the subfix ";-n" as there
                # token.
                # NOTE: each non-keyframe cam is associated with its next
                # keyframe sample.
                info['token'] = info['token'] + f";-{frame_cnt}"
                infos.append(info)
                cams = _get_pn_cams(nusc, direction='prev', cams=cams)
                frame_cnt += 1

        # construct scene tokens, prev should be reversed
        if (len(infos) == frames_t or len(infos) == 1) or not use_break or not with_cam_sweeps:
            __scene_tokens = [info['token'] for info in infos]
            __scene_tokens.reverse()
            _scene_tokens += __scene_tokens
        else:
            breaks += 1
            # this clip does not have `frames_t` frames
            mid_head = infos[0]['token']

        prev_sample = sample

        if sample["scene_token"] in train_scenes:
            train_nusc_infos += infos
        else:
            val_nusc_infos += infos

    # finally, we add the last one if exists.
    if len(_scene_tokens) > frames_t:
        if scene_token in train_scenes:
            train_scene_tokens.append(_scene_tokens)
        else:
            val_scene_tokens.append(_scene_tokens)

    print(f"Total break scenes: {breaks}")
    print(f"Total train scenes: {len(train_scene_tokens)}")
    print(f"Total val scenes: {len(val_scene_tokens)}")
    return train_nusc_infos, train_scene_tokens, val_nusc_infos, val_scene_tokens


def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: True.
    """
    # get bbox annotations for camera
    nusc_infos = mmcv.load(info_path)["infos"]
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info["cams"][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info["sample_data_token"],
                visibilities=["", "1", "2", "3", "4"],
                mono3d=mono3d,
            )
            (height, width, _) = mmcv.imread(cam_info["data_path"]).shape
            coco_2d_dict["images"].append(
                dict(
                    file_name=cam_info["data_path"].split("data/nuscenes/")[-1],
                    id=cam_info["sample_data_token"],
                    token=info["token"],
                    cam2ego_rotation=cam_info["sensor2ego_rotation"],
                    cam2ego_translation=cam_info["sensor2ego_translation"],
                    ego2global_rotation=info["ego2global_rotation"],
                    ego2global_translation=info["ego2global_translation"],
                    camera_intrinsics=cam_info["camera_intrinsics"],
                    width=width,
                    height=height,
                )
            )
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info["segmentation"] = []
                coco_info["id"] = coco_ann_id
                coco_2d_dict["annotations"].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f"{info_path[:-4]}_mono3d"
    else:
        json_prefix = f"{info_path[:-4]}"
    mmcv.dump(coco_2d_dict, f"{json_prefix}.coco.json")


def get_2d_boxes(
        nusc, sample_data_token: str, visibilities: List[str],
        mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get("sample_data", sample_data_token)

    assert sd_rec["sensor_modality"] == "camera", (
        "Error: get_2d_boxes only works" " for camera sample_data!"
    )
    if not sd_rec["is_key_frame"]:
        raise ValueError(
            "The 2D re-projections are available only for keyframes.")

    s_rec = nusc.get("sample", sd_rec["sample_token"])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    camera_intrinsic = np.array(cs_rec["camera_intrinsic"])

    # Get all the annotation with the specified visibilties.
    ann_recs = [nusc.get("sample_annotation", token) for token in s_rec["anns"]]
    ann_recs = [ann_rec for ann_rec in ann_recs if (
        ann_rec["visibility_token"] in visibilities)]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec["sample_annotation_token"] = ann_rec["token"]
        ann_rec["sample_data_token"] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec["token"])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec["translation"]))
        box.rotate(Quaternion(pose_rec["rotation"]).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec["translation"]))
        box.rotate(Quaternion(cs_rec["rotation"]).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = (
            view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()
        )

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(
            ann_rec, min_x, min_y, max_x, max_y, sample_data_token,
            sd_rec["filename"])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec["rotation"]).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec["rotation"]).rotation_matrix
            cam_velo3d = (global_velo3d @ np.linalg.inv(e2g_r_mat).T @
                          np.linalg.inv(c2e_r_mat).T)
            velo = cam_velo3d[0::2].tolist()

            repro_rec["bbox_cam3d"] = loc + dim + rot
            repro_rec["velo_cam3d"] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec["center2d"] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec["center2d"][2] <= 0:
                continue

            ann_token = nusc.get("sample_annotation", box.token)[
                "attribute_tokens"]
            if len(ann_token) == 0:
                attr_name = "None"
            else:
                attr_name = nusc.get("attribute", ann_token[0])["name"]
            attr_id = nus_attributes.index(attr_name)
            repro_rec["attribute_name"] = attr_name
            repro_rec["attribute_id"] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords]
        )

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(
    ann_rec: dict,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    sample_data_token: str,
    filename: str,
) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec["sample_data_token"] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        "attribute_tokens",
        "category_name",
        "instance_token",
        "next",
        "num_lidar_pts",
        "num_radar_pts",
        "prev",
        "sample_annotation_token",
        "sample_data_token",
        "visibility_token",
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec["bbox_corners"] = [x1, y1, x2, y2]
    repro_rec["filename"] = filename

    coco_rec["file_name"] = filename
    coco_rec["image_id"] = sample_data_token
    coco_rec["area"] = (y2 - y1) * (x2 - x1)

    if repro_rec["category_name"] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec["category_name"]]
    coco_rec["category_name"] = cat_name
    coco_rec["category_id"] = nus_categories.index(cat_name)
    coco_rec["bbox"] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec["iscrowd"] = 0

    return coco_rec
