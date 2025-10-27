__all__ = []

import torch
def torch_nn_Module_ve(self, device=None):
	return self.to(device=torch.device('ve', device) if device == None or isinstance(device, int) else device)

torch.nn.Module.ve = torch_nn_Module_ve
