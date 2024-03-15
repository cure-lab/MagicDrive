from typing import List, Dict, Any
import pickle
import importlib
from copy import deepcopy
from functools import update_wrapper

import torch
import accelerate
from accelerate.state import AcceleratorState
from accelerate.utils import recursively_apply


def unsqueeze_tensors_in_dict(in_dict: Dict[str, Any], dim) -> Dict[str, Any]:
    out_dict = {}
    for k, v in in_dict.items():
        if isinstance(v, torch.Tensor):
            out_dict[k] = v.unsqueeze(dim)
        elif isinstance(v, dict):
            out_dict[k] = unsqueeze_tensors_in_dict(v, dim)
        elif isinstance(v, list):
            if dim == 0:
                out_dict[k] = [v]
            elif dim == 1:
                out_dict[k] = [[vi] for vi in v]
            else:
                raise ValueError(
                    f"cannot handle {k}:{v} ({v.__class__}) with dim={dim}")
        elif v is None:
            out_dict[k] = None
        else:
            raise TypeError(f"Unknow dtype for {k}:{v} ({v.__class__})")
    return out_dict


def stack_tensors_in_dicts(
        dicts: List[Dict[str, Any]], dim, holder=None) -> Dict[str, Any]:
    """stack any Tensor in list of dicts. If holder is provided, dicts will be
    stacked ahead of holder tensor. Make sure no dict is changed in place.

    Args:
        dicts (List[Dict[str, Any]]): dicts to stack, without the desired dim.
        dim (int): dim to add for stack.
        holder (_type_, optional): dict to hold, with the desired dim. Defaults
        to None. 

    Raises:
        TypeError: if the datatype for values are not Tensor or dict.

    Returns:
        Dict[str, Any]: stacked dict.
    """
    if len(dicts) == 1:
        if holder is None:
            return unsqueeze_tensors_in_dict(dicts[0], dim)
        else:
            this_dict = dicts[0]
            final_dict = deepcopy(holder)
    else:
        this_dict = dicts[0]  # without dim
        final_dict = stack_tensors_in_dicts(dicts[1:], dim)  # with dim
    for k, v in final_dict.items():
        if isinstance(v, torch.Tensor):
            # for v in this_dict, we need to add dim before concat.
            if this_dict[k].shape != v.shape[1:]:
                print("Error")
            final_dict[k] = torch.cat([this_dict[k].unsqueeze(dim), v], dim=dim)
        elif isinstance(v, dict):
            final_dict[k] = stack_tensors_in_dicts(
                [this_dict[k]], dim, holder=v)
        elif isinstance(v, list):
            if dim == 0:
                final_dict[k] = [this_dict[k]] + v
            elif dim == 1:
                final_dict[k] = [
                    [this_vi] + vi for this_vi, vi in zip(this_dict[k], v)]
            else:
                raise ValueError(
                    f"cannot handle {k}:{v} ({v.__class__}) with dim={dim}")
        elif v is None:
            assert final_dict[k] is None
        else:
            raise TypeError(f"Unknow dtype for {k}:{v} ({v.__class__})")
    return final_dict


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
