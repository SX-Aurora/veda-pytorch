__all__ = []

import torch
def ve_tag(obj):
	if obj.device.type == 've':
		return 've'
	
def ve_deserialize(obj, location):
	if location.startswith('ve'):
		return obj.ve(location)
torch.serialization.register_package(11, ve_tag, ve_deserialize)