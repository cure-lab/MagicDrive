import pickle
import importlib
from functools import update_wrapper

import torch
import accelerate
from accelerate.state import AcceleratorState
from accelerate.utils import recursively_apply


def load_module(name):
    p, m = name.rsplit(".", 1)
    mod = importlib.import_module(p)
    model_cls = getattr(mod, m)
    return model_cls


def move_to(obj, device, filter=lambda x: True):
    if torch.is_tensor(obj):
        if filter(obj):
            return obj.to(device)
        else:
            return obj
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device, filter)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device, filter))
        return res
    elif obj is None:
        return obj
    else:
        raise TypeError(f"Invalid type {obj.__class__} for move_to.")


# take from torch.ao.quantization.fuse_modules
# Generalization of getattr
def _get_module(model, submodule_key):
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


# Generalization of setattr
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)

    setattr(cur_mod, tokens[-1], module)


def convert_to_fp16(tensor):
    """
    Recursively converts the elements nested list/tuple/dictionary of tensors in FP32 precision to FP16.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to convert from FP32 to FP16.

    Returns:
        The same data structure as `tensor` with all tensors that were in FP32 precision converted to FP16.
    """

    def _convert_to_fp16(tensor):
        return tensor.half()

    def _is_fp32_tensor(tensor):
        return hasattr(tensor, "dtype") and (
            tensor.dtype == torch.float32
        )

    return recursively_apply(_convert_to_fp16, tensor,
                             test_type=_is_fp32_tensor)


class ConvertOutputsToFp16:
    """
    Decorator to apply to a function outputing tensors (like a model forward pass) that ensures the outputs in FP32
    precision will be convert back to FP16.

    Args:
        model_forward (`Callable`):
            The function which outputs we want to treat.

    Returns:
        The same function as `model_forward` but with converted outputs.
    """

    def __init__(self, model_forward):
        self.model_forward = model_forward
        update_wrapper(self, model_forward)

    def __call__(self, *args, **kwargs):
        return convert_to_fp16(self.model_forward(*args, **kwargs))

    def __getstate__(self):
        raise pickle.PicklingError(
            "Cannot pickle a prepared model with automatic mixed precision, please unwrap the model with `Accelerator.unwrap_model(model)` before pickling it."
        )


convert_outputs_to_fp16 = ConvertOutputsToFp16


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin \
        if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
