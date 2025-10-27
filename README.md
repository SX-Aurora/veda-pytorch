# VEDA PyTorch

VEDA PyTorch is a library to add device support for the NEC SX-Aurora TSUBASA
into PyTorch.

[![Github](https://img.shields.io/github/v/tag/sx-aurora/veda-pytorch?display_name=tag&sort=semver)](https://github.com/sx-aurora/veda)
[![PyPI](https://img.shields.io/pypi/v/veda-pytorch)](https://pypi.org/project/veda-pytorch)
[![License](https://img.shields.io/pypi/l/veda-pytorch)](https://pypi.org/project/veda-pytorch)
![Python Versions](https://img.shields.io/pypi/pyversions/veda-pytorch)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
![Maintenance](https://img.shields.io/pypi/dm/veda-pytorch)

## Release Notes
<table>
<tr><th>Version</th><th>Comment</th></tr>

<tr><td>v14</td><td>
Starting with v14, VEDA PyTorch is no longer distributed as precompiled binary
but gets compiled as PyTorch C++ extension on the target machine. So you don't
need to install a matching binary package anymore!

We further added a experimental implementation for using NEC MPI. You can create
the process group as follows:

<code>
torch.distributed.init_process_group(
    backend		= 'veda',
    world_size	= os.environ['MPISIZE'],
    rank		= os.environ['MPIRANK'],
	store		= torch.distributed.Store()
)
</code>

Further changes:
<ul>
	<li>Added support for PyTorch v2.9.0</li>
	<li>Added <code>arange.start_out</code></li>
	<li>Added function tracing. Activate using <code>TUNGL_LOG=TRACE</code></li>
	<li>Bugfix for <code>aten::cat.out</code></li>
	<li>Bugfix for <code>copy_</code></li>
	<li>Bugfix for <code>torch.load(location='ve')</code></li>
	<li>Removed unnecessary context sync</li>
</ul>
</td></tr>

<tr><td>v13</td><td>
<ul>
	<li>Fixed <code>torch.ve.set_device</code></li>
	<li>Fixed allocation on wrong VE in multi-process execution</li>
	<li>Improved error messages</li>
	<li>Upgraded build script for PyTorch >=2.7!</li>
</ul>
</td></tr>

<tr><td>v12</td><td>
<ul>
	<li>Added auto plugin loading for Pytorch. <code>import veda.pytorch</code> is no longer required with PyTorch >=2.5!</li>
</ul>
</td></tr>

<tr><td>v11</td><td>
<ul>
	<li>Fixed shutdown problem in mixed GPU/VE use cases.</li>
</ul>
</td></tr>

<tr><td>v10</td><td>
<ul>
	<li>Support for PyTorch v2.3.1</li>
	<li>Support for SX-Aurora VE3</li>
</ul>
</td></tr>

<tr><td>v9</td><td>
<ul>
	<li>Support for PyTorch v2.3.0</li>
</ul>
</td></tr>

<tr><td>v8</td><td>
<ul>
	<li>Added <code>torch.logical_not</code></li>
</ul>
</td></tr>

<tr><td>v7</td><td>
<ul>
	<li>Support for PyTorch v2.0.0</li>
	<li>Support for PyTorch v1.13.0</li>
	<li>Added <code>torch.log1p</code></li>
</ul>
</td></tr>

<tr><td>v6</td><td>
<ul>
	<li>Support for PyTorch v1.12.0 and v1.12.1</li>
</ul>
</td></tr>

<tr><td>v5</td><td>
<ul>
	<li>Added <ul>
		<li><code>torch.clamp</code></li>
		<li><code>torch.clamp_max</code></li>
		<li><code>torch.clamp_min</code></li>
		<li><code>torch.exp</code></li>
		<li><code>torch.log</code></li>
		<li><code>torch.norm</code></li>
		<li><code>torch.pow</code></li>
		<li><code>torch.where</code></li>
	</ul></li>
	<li>Fixed conversion from numeric value to bool</li>
	<li>Fixed calling <code>torch.ve.memory_allocated()</code> without device id</li>
	<li>Preventing 0-byte allocations from PyTorch to be passed on to VEDA</li>
</ul>
</td></tr>

<tr><td>v4</td><td>
<ul>
	<li>fixed possible segfault in Tensor resize if no storage is initialized</li>
	<li>fixed dtype handling in Scalar to Tensor operations</li>
</ul>
</td></tr>

<tr><td>v3</td><td>
<ul>
	<li>added squeeze and unsqueeze handlers</li>
</ul>
</td></tr>

<tr><td>v2</td><td>
<ul>
	<li>Minor changes to enable PyTorch v1.11.0</li>
	<li>Fixed vedaInit error checking to ignore if already initialized</li>
</ul>
</td></tr>

<tr><td>v1</td><td>
Initial Release
</td></tr>

</table>
