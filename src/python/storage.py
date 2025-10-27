__all__ = []

import torch

def torch_UntypedStorage_ve(self, device=None):
	untyped_storage = torch.empty(self.nbytes(), dtype=torch.uint8, device=device, pin_memory=False).untyped_storage()
	untyped_storage.copy_(self, False)
	return untyped_storage

torch.UntypedStorage.ve = torch_UntypedStorage_ve