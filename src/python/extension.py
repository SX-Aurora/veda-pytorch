__all__ = []

from packaging.version import Version
from pathlib import Path
import torch
import torch.utils.cpp_extension
import tungl

sources				= []
include_directories	= []
ldflags				= ['-Wl,--disable-new-dtags']

def find_sources(path = ''):
	src_path = (Path(__file__).parent / 'csrc'/ 'veda' / 'pytorch' / path)
	return [str(file) for file in src_path.glob('*.cpp')]

# Tungl ------------------------------------------------------------------------
tungl_dir			= Path(tungl.__file__).parent
tungl_lib			= tungl_dir / 'lib64'
include_directories += [str(tungl_dir / 'include')]
ldflags				+= ['-L' + str(tungl_lib), '-ltungl', '-Wl,-rpath=' + str(tungl_lib)]

# VEDA -------------------------------------------------------------------------
veda_dir			= Path(tungl_dir).parent / 'veda'
veda_lib			= veda_dir / 'lib64'
include_directories	+= [str(veda_dir / 'include')]
ldflags				+= ['-L' + str(veda_lib), '-lveda', '-Wl,-rpath=' + str(veda_lib)]

# VEDA-Tensors -----------------------------------------------------------------
veda_tensors_lib	= veda_dir / 'tensors' / 'lib64'
include_directories	+= [str(veda_dir / 'tensors' / 'include')]
ldflags				+= ['-L' + str(veda_tensors_lib), '-lveda-tensors', '-Wl,-rpath=' + str(veda_tensors_lib)]

# ATEN -------------------------------------------------------------------------
sources += find_sources()

# MPI --------------------------------------------------------------------------
def find_mpi():
	global sources
	global include_directories
	global ldflags
	mpi_root	= Path('/opt/nec/ve/mpi')
	mpi_version = Version('0')
	for path in mpi_root.glob('*'):
		if path.name != 'libexec':
			version = Version(path.name)
			if mpi_version < version:
				mpi_version = version

	if mpi_version == Version('0'):
		return None
	
	mpi_path			= mpi_root / str(mpi_version)
	mpi_lib				= mpi_path / 'lib64' / 'vh' / 'gnu' / 'default'
	sources				+= find_sources('c10d')
	include_directories += [str(mpi_path / 'include')]
	ldflags				+= ['-L' + str(mpi_lib), '-Wl,-rpath=' + str(mpi_lib)]
	ldflags				+= ['-lmpi', '-lmpi++', '-lmpi_veo', '-lgfortran', str(mpi_lib / 'mpir_dummy_gfc_PIC.o')]
	return mpi_version

mpi_version = find_mpi()

# Compile ----------------------------------------------------------------------
torch.utils.cpp_extension.load(
	'veda_pytorch._C',
	sources,
	extra_ldflags		= ldflags,
	extra_include_paths	= include_directories,
	keep_intermediates	= False,
)

# Register ProcessGroup --------------------------------------------------------
from torch._C._distributed_c10d import ProcessGroupVEDA
torch.distributed.distributed_c10d.Backend.register_backend("veda", lambda dist_backend_opts, backend_options: ProcessGroupVEDA.create(dist_backend_opts), True, ['ve'])