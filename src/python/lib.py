__all__ = ['autoload']

import ctypes
import os
import collections
import torch # make sure pytorch was loaded before

if not "@PYTORCH_VERSION@" in torch.__version__:
	raise Exception(f"The NEC SX-Aurora TSUBASA can only be used with PyTorch v@PYTORCH_VERSION@ but you are using {torch.__version__}")

lib	= ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'lib64/libveda-pytorch.so'))
lib.veda_pytorch_get_current_device.argtypes	= []
lib.veda_pytorch_get_current_device.restype		= ctypes.c_int
lib.veda_pytorch_device_count.argtypes			= []
lib.veda_pytorch_device_count.restype			= ctypes.c_int
lib.veda_pytorch_set_device.argtypes			= [ctypes.c_int]
lib.veda_pytorch_set_device.restype				= None
lib.veda_pytorch_memory_allocated.argtypes		= [ctypes.c_int]
lib.veda_pytorch_memory_allocated.restype		= ctypes.c_long
lib.veda_pytorch_sync.argtypes					= [ctypes.c_int]
lib.veda_pytorch_sync.restype					= None

def autoload(): # nothing special to do
	pass

def get_device_idx(device):
	if device is None:
		return lib.veda_pytorch_get_current_device()
	
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
	return torch.device(f've:{get_device_idx(None)}')

def set_device(device):
	lib.veda_pytorch_set_device(get_device_idx(device))

def device_count():
	return int(lib.veda_pytorch_device_count())

def is_available():
	return device_count() > 0

def memory_allocated(device=None):
	return int(lib.veda_pytorch_memory_allocated(get_device_idx(device)))

def to_ve(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
	assert device == None or isinstance(device, int)
	return self.to(device=torch.device('ve', device), non_blocking=non_blocking, memory_format=memory_format)

def to_model_ve(self, device=None):
	assert device == None or isinstance(device, int)
	return self.to(device=torch.device('ve', device))

class Device:
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
		
class DeviceOf(Device):
	def __init__(self, tensor):
		super().__init__(tensor.device)

def synchronize(device=None):
	if device is None:
		for i in range(device_count()):
			lib.veda_pytorch_sync(i)
	else:
		lib.veda_pytorch_sync(get_device_idx(device))

torch.ve = collections.namedtuple('VE',
	[
		'synchronize',
		'is_available',
		'current_device',
		'set_device',
		'device_count',
		'device',
		'device_of',
		'memory_allocated'
	])(
		synchronize, 
		is_available,
		current_device,
		set_device,
		device_count,
		Device,
		DeviceOf,
		memory_allocated
	)
torch.Tensor.ve		= to_ve
torch.nn.Module.ve	= to_model_ve