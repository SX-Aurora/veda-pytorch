__all__ = []

import torch

def torch_Tensor_ve(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
	return self.to(device=torch.device('ve', device) if device == None or isinstance(device, int) else device, non_blocking=non_blocking, memory_format=memory_format)

torch.Tensor.ve	= torch_Tensor_ve
