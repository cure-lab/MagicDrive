#############################
# interp
############################

# create ann_info
python tools/create_data.py nuscenes --root-path ../data/nuscenes --out-dir ../data/nuscenes_mmdet3d-12Hz --extra-tag nuscenes_interp_12Hz --max-sweeps -1 --version interp_12Hz_trainval

# "nuscenes_interp_12Hz_dbinfos_train.pkl" can be moved to `../data/nuscenes_mmdet3d-12Hz/`

# generate map cache for val
python tools/prepare_map_aux.py +exp=map_cache_gen_interp +process=val +subfix=12Hz_interp

# generate map cache for train
python tools/prepare_map_aux.py +exp=map_cache_gen_interp +process=train +subfix=12Hz_interp

# then move cache files to `../data/nuscenes_map_aux_12Hz_interp`

#############################
# advanced
############################

# create ann_info
python tools/create_data.py nuscenes --root-path ../data/nuscenes --out-dir ../data/nuscenes_mmdet3d-12Hz --extra-tag nuscenes_advanced_12Hz --max-sweeps -1 --version advanced_12Hz_trainval

# "nuscenes_advanced_12Hz_dbinfos_train.pkl" can be moved to `../data/nuscenes_mmdet3d-12Hz/`

# generate map cache for val
python tools/prepare_map_aux.py +exp=map_cache_gen_advanced +process=val +subfix=12Hz_advanced

# generate map cache for train
python tools/prepare_map_aux.py +exp=map_cache_gen_advanced +process=train +subfix=12Hz_advanced

# then move cache files to `../data/nuscenes_map_aux_12Hz_adv`
