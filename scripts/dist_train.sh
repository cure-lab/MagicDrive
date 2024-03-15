NUM_PROCESSES=$1
PY_ARGS=${@:2}
LAUNCH_PARAM=${LAUNCH_PARAM:-"--num_processes ${NUM_PROCESSES} --num_machines 1"}

accelerate launch --mixed_precision fp16 --gpu_ids all \
    ${LAUNCH_PARAM} tools/train.py ${PY_ARGS}
