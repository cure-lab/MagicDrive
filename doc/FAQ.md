### Q1: [Error] AttributeError: module 'distutils' has no attribute 'version'

Fix:

```
pip install setuptools==59.5.0
```

Ref: [pytorch #69894](https://github.com/pytorch/pytorch/issues/69894#issuecomment-1080635462)

### Q2: How to setup on Ampere GPUs (e.g., A100, A800)?

As informed [here](https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version), please pick a compatible gcc version and cuda version. Note that, you should use cuda 11+ for Ampere GPUs.

We tested with `python=3.8`, `gcc-0.9.3`, `g++-0.9.3`, and `cuda-11.3`. The final environment file can be find [requirements/a800.yaml](../requirements/a800.yaml), where `mmdet3d` from `third_party/bevfusion` is not included since we install it with `develop` mode.

Otherwise, here is a step-by-step guide to setup the environment above.

```bash
# python3.8, cuda113
pip install https://download.pytorch.org/whl/cu113/torch-1.10.2%2Bcu113-cp38-cp38-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu113/torchvision-0.11.3%2Bcu113-cp38-cp38-linux_x86_64.whl
pip install https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/mmcv_full-1.4.5-cp38-cp38-manylinux1_x86_64.whl

# now, you need to comment torch, torchvision, and mmcv_full in 
# `requirements/dev.txt`, and run
pip install -r requirements/dev.txt

cd third_party/diffusers
pip install .

cd third_party/bevfusion
python setup.py develop
```

Now, you should be able to run our demo.

### Q3: [Error] nvcc fatal   : Unsupported gpu architecture 'compute_80'

This may appear when you install bevfusion (mmdet3d) on cuda10.2. The latest version of bevfusion supports Ampere GPUs by hard-coding compile parameters, leading to error when compiled with cuda10.2. One can get rid of this error by comment these lines in `third_party/bevfusion/setup.py (L19)`.
```python
"-D__CUDA_NO_HALF2_OPERATORS__",
"-gencode=arch=compute_70,code=sm_70",
"-gencode=arch=compute_75,code=sm_75",
# "-gencode=arch=compute_80,code=sm_80",
# "-gencode=arch=compute_86,code=sm_86",
```

### Q4: video generation

The authors do not have plan to integrate video generation with `main` branch codebase. For those who need code for video generation, please find this in `video` branch.


### Q5: AttributeError: 'BasicTransformerBlock' object has no attribute '_args'

[#7](https://github.com/cure-lab/MagicDrive/issues/7).
This can occur when you install the official version of diffusers rather than from `third_party/diffusers`.
To install the local version, please make sure you use the following commands:
```bash
# if you have install, remove first
pip uninstall diffusers

# install our version
cd third_party/diffusers
pip install -vvv .
```


### Q6: model loading path vs `accelerate` checkpoints
[#25](https://github.com/cure-lab/MagicDrive/issues/25).
There are two formats for model saving.
- `checkpoints-xxxx`: saved by `accelerate`, default by `runner.checkpointing_steps: 5000`. This is used to resume training process.
- `weight-Exx-Sxxxxx`: saved by `diffusers`, default is on the end of each epoch/model training. This is used to load for testing and evaluation.

If you want to change the frequency for saving `diffusers` format for testing, please invoke [this line](https://github.com/cure-lab/MagicDrive/blob/0874565a8c3d35c7a3af0c3f78a7e664460e2e8b/magicdrive/runner/base_runner.py#L381C30-L381C41) manual in your code.
