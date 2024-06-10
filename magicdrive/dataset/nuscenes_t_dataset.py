import logging

import mmcv
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset


@DATASETS.register_module()
class NuScenesTDataset(NuScenesDataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length=None,
        start_on_keyframe=True,
    ) -> None:
        self.video_length = video_length
        self.start_on_keyframe = start_on_keyframe
        super().__init__(
            ann_file, pipeline, dataset_root, object_classes, map_classes,
            load_interval, with_velocity, modality, box_type_3d,
            filter_empty_gt, test_mode, eval_version, use_valid_flag,
            force_all_boxes)
        if "12Hz" in ann_file and start_on_keyframe:
            logging.warn("12Hz should use all starting frame to train, please"
                         "double-check!")

    def build_clips(self, data_infos, scene_tokens):
        """Since the order in self.data_infos may change on loading, we
        calculate the index for clips after loading.

        Args:
            data_infos (list of dict): loaded data_infos
            scene_tokens (2-dim list of str): 2-dim list for tokens to each
            scene 

        Returns:
            2-dim list of int: int is the index in self.data_infos
        """
        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)}
        all_clips = []
        for scene in scene_tokens:
            for start in range(len(scene) - self.video_length + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start] >= 33):
                    continue  # this is not a keyframe
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + self.video_length]]
                all_clips.append(clip)
        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Cut into {self.video_length}-clip, "
                     f"which has {len(all_clips)} in total.")
        return all_clips

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        self.clip_infos = self.build_clips(data_infos, data['scene_tokens'])
        return data_infos

    def __len__(self):
        return len(self.clip_infos)

    def get_data_info(self, index):
        """We should sample from clip_infos
        """
        clip = self.clip_infos[index]
        frames = []
        for frame in clip:
            frame_info = super().get_data_info(frame)
            info = self.data_infos[frame]
            frames.append(frame_info)
        return frames

    def prepare_train_data(self, index):
        """This is called by `__getitem__`
        """
        frames = self.get_data_info(index)
        if None in frames:
            return None
        examples = []
        for frame in frames:
            self.pre_pipeline(frame)
            example = self.pipeline(frame)
            if self.filter_empty_gt and frame['is_key_frame'] and (
                example is None or ~(example["gt_labels_3d"]._data != -1).any()
            ):
                return None
            examples.append(example)
        return examples
