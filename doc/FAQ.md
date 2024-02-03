### Q1: [Error] AttributeError: module 'distutils' has no attribute 'version'

Fix:

```
pip install setuptools==59.5.0
```

Ref: [pytorch #69894](https://github.com/pytorch/pytorch/issues/69894#issuecomment-1080635462)

### Q2: How to setup on Ampere GPUs (e.g., A100, A800)?

As informed [here](https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version), please pick a compatible gcc version and cuda version. Note that, you should use cuda 11+ for Ampere GPUs.

We tested with `python=3.8`, `gcc-0.9.3`, `g++-0.9.3`, and `cuda-11.3`. The final environment file can be find [requirements/a800.yaml](requirements/a800.yaml), where `mmdet3d` from `third_party/bevfusion` is not included since we install it with `develop` mode.

Otherwise, here is a step-by-step guide to setup the environment above.

```bash
# python3.8, cuda113
pip install https://download.pytorch.org/whl/cu113/torch-1.10.2%2Bcu113-cp38-cp38-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu113/torchvision-0.11.3%2Bcu113-cp38-cp38-linux_x86_64.whl
pip install https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/mmcv_full-1.4.5-cp38-cp38-manylinux1_x86_64.whl

# now, you need to comment torch, torchvision, and mmcv_full in 
# `requirements/dev.txt`, and run
pip install -r requirements/dev.txt

cd third_party
pip install diffusers

cd third_party/bevfusion
python setup.py develop
```

Now, you should be able to run our demo.
