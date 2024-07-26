import random
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes


def sample_token_from_scene(ratio_or_num, nusc=None, drop_desc=None):
    """Sample keyframes from each scene according to ratio.
    if ratio_or_num >= 1, treated as num;
    if ratio_or_num < 1, treated as ratio;
    if ratio_or_num == 0, only pick the first frame;
    if ratio_or_num == -1, return None.

    Args:
        ratio (float): sample ratio to each scene.

    Returns:
        sample_flag_dict (dict): Dict[token, bool]
        scene_sample_flag_dict (dict): Dict[scene_name, Dict[token, bool]]
    """
    if ratio_or_num == -1 and drop_desc is None:
        return None, None
    if nusc is None:
        nusc = NuScenes(version='v1.0-trainval',
                        dataroot='./data/nuscenes', verbose=True)
    sample_flag_dict = {}
    scene_sample_flag_dict = {}
    for idx, scene in enumerate(nusc.scene):
        scene_name = scene['name']
        frames_len = scene['nbr_samples']
        sample_token = scene['first_sample_token']
        # iteratively gather sample tokens from one scene
        token_in_this_scene = OrderedDict()
        for fi in range(frames_len):
            token_in_this_scene[sample_token] = False
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']
        desc = scene['description']
        if drop_desc is not None and drop_desc in desc.lower():
            picked = []  # we pick nothing
        else:
            # pick tokens according to your ratio
            if ratio_or_num == 0:
                # if 0, only pick the first one
                picked = list(token_in_this_scene.keys())[0:1]
            else:
                if ratio_or_num >= 1:
                    pick_num = int(ratio_or_num)
                else:
                    pick_num = int(frames_len * ratio_or_num)
                picked = random.sample(token_in_this_scene.keys(), pick_num)
        for pick in picked:
            token_in_this_scene[pick] = True
        # now save data for output
        token_in_this_scene = dict(token_in_this_scene)
        scene_sample_flag_dict[scene_name] = token_in_this_scene
        sample_flag_dict.update(token_in_this_scene)
    return sample_flag_dict, scene_sample_flag_dict
