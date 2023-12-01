import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist


class DistributedModule(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.module.register_forward_pre_hook(self._allreduce_parameters)
        self.module.register_state_dict_pre_hook(self._wait_all_handles)
        # for p in self._parameters:
        #    p.register_hook()
        self._handles = []

    def _allreduce_parameters(self, module, args):
        handle = dist(
            self.module.parameters(), op=dist.ReduceOp.AVG, async_op=True
        )
        self._handles.append(handle)

    def _wait_all_handles(self, *args, **kwargs):
        for handle in self._handles:
            handle.wait()
        self._handles.clear()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
