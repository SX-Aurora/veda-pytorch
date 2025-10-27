__all__ = [
	'synchronize',
	'is_available',
	'current_device',
	'set_device',
	'device_count',
	'device',
	'device_of',
	'memory_allocated',
	'get_amp_supported_dtype',
	'manual_seed_all',
	'_is_in_bad_fork',
]

import torch

def _get_device_idx(device):
	if device is None:
		return torch.ops.veda.get_current_device()
	
	if isinstance(device, str):
		device = torch.device(device)

	if isinstance(device, torch.device):
		assert device.type == 've', f"Expected device to be 've' but is '{device}'"
		device = device.index
	
	if isinstance(device, int):
		assert device >= 0 and device < device_count(), f"Expected device index to be >= 0 and < {device_count()} but is {device}"
		return device
	
	raise Exception(f"Invalid type, expected None, str, torch.device or int but is {type(device)}")

def current_device():
	return torch.device(f've:{_get_device_idx(None)}')

def set_device(device):
	torch.ops.veda.set_device(_get_device_idx(device))

def device_count():
	return torch.ops.veda.device_count()

def is_available():
	return device_count() > 0

def memory_allocated(device=None):
	return torch.ops.veda.memory_allocated(_get_device_idx(device))

class device:
	def __init__(self, device):
		if isinstance(device, str):
			device = torch.device(device)
			assert device.type == 've'
		assert isinstance(device, torch.device) or isinstance(device, int)
		self.m_previous = None
		self.m_device   = device.index if isinstance(device, torch.device) else device

	def __enter__(self):
		assert isinstance(self.m_device, int)
		self.m_previous = current_device()
		set_device(self.m_device)
		return self

	def __exit__(self, type, value, traceback):
		assert isinstance(self.m_previous, torch.device)
		set_device(self.m_previous)
		
class device_of(device):
	def __init__(self, tensor):
		super().__init__(tensor.device)

def synchronize(device=None):
	if device is None:	torch.ops.veda.sync_all()
	else:				torch.ops.veda.sync(_get_device_idx(device))

def get_amp_supported_dtype():
	return [torch.float, torch.double]

def manual_seed_all(seed: int) -> None:
	pass

def _is_in_bad_fork():
	return False

import sys
torch.ve				= sys.modules[__name__]
sys.modules['torch.ve']	= torch.ve
