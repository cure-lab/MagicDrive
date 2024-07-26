import torch
from accelerate import Accelerator


def concat_from_everyone(accelerator: Accelerator, tmp):
    if not accelerator.use_distributed:
        return tmp
    output = [None for _ in range(accelerator.num_processes)]
    torch.distributed.all_gather_object(output, tmp)
    if accelerator.is_main_process:
        res = []
        for tmpi in output:
            res += tmpi
        return res
    else:
        return None
