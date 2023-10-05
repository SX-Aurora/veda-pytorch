# VEDA PyTorch

VEDA PyTorch is a library to add device support for the NEC SX-Aurora TSUBASA into PyTorch.

[![Github](https://img.shields.io/github/v/tag/sx-aurora/veda-pytorch?display_name=tag&sort=semver)](https://github.com/sx-aurora/veda)
[![PyPI](https://img.shields.io/pypi/v/veda-pytorch)](https://pypi.org/project/veda-pytorch)
[![License](https://img.shields.io/pypi/l/veda-pytorch)](https://pypi.org/project/veda-pytorch)
![Python Versions](https://img.shields.io/pypi/pyversions/veda-pytorch)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
![Maintenance](https://img.shields.io/pypi/dm/veda-pytorch)

## Release Notes
<table>
<tr><th>Version</th><th>Comment</th></tr>

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
