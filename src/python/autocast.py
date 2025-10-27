__all__ = []

import torch
import tungl

try:
	torch_amp_is_autocast_available = torch.amp.is_autocast_available
	def is_autocast_available(device_type: str) -> bool:
		if device_type.startswith('ve'):
			return True
		return torch_amp_is_autocast_available(device_type)
	torch.amp.autocast_mode.is_autocast_available = is_autocast_available

	torch_get_autocast_dtype = torch.get_autocast_dtype
	def get_autocast_dtype(device_type: str) -> torch.dtype:
		if device_type.startswith('ve'):
			return torch.float32
		return torch_get_autocast_dtype(device_type)
	torch.get_autocast_dtype = get_autocast_dtype

	torch_is_autocast_enabled = torch.is_autocast_enabled
	def is_autocast_enabled(device_type: str = None) -> bool:
		if device_type is not None:
			if device_type.startswith('ve'):
				return False
			return torch_is_autocast_enabled(device_type)
		return torch_is_autocast_enabled()
	torch.is_autocast_enabled = is_autocast_enabled

	torch_set_autocast_enabled = torch.set_autocast_enabled
	def set_autocast_enabled(device: str, enabled: bool):
		if device.startswith('ve'):
			return
		return torch_set_autocast_enabled(device, enabled)
	torch.set_autocast_enabled = set_autocast_enabled
except:
	tungl.error("Unable to integrate VEDA PyTorch into torch.autocast. Please file a bug report!", module="VEDA-PyTorch")
