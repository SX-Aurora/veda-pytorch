try:
	from .extension import *
except ImportError as e:
	import tungl
	tungl.error("Unable to load extension: ", str(e), module="VEDA-PyTorch")
except RuntimeError as e:
	import tungl
	import torch
	tungl.error(f"Unable to compile extension. Maybe your PyTorch version ({torch.__version__}) is not supported:", str(e))
finally:
	from .ve			import *	# torch.ve
	from .autocast		import *	# torch.autocast
	from .tensor		import *	# torch.Tensor.ve
	from .nn_module		import *	# torch.nn.Module.ve
	from .storage		import *	# torch.UntypedStorage.ve
	from .serialization	import *	# torch.serialization
	from .autoload		import *