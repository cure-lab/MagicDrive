#!/bin/bash

CKPT=${1%/}  # e.g., magicdrive-t-log/StableDiffusion-v1.5_2023-05-15_15-14-33_0.0.6/
ARGS=${@:2}  # e.g., show_box=False (to disable box drawing)

TAG_SUBFIX=${TAG_SUBFIX:-""}

# get tab, e.g., 0.0.6
readarray -d _ -t TAG < <(printf '%s' "${CKPT}")
TAG="${TAG[-1]}${TAG_SUBFIX}"
echo Run for ${TAG}

set -x

python tools/test.py \
    resume_from_checkpoint=${CKPT} task_id=${TAG} ${ARGS}
