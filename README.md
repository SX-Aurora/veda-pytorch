# VEDA PyTorch

VEDA PyTorch is a library to add device support for the NEC SX-Aurora TSUBASA into PyTorch.

## Release Notes
<table>
<tr><th>Version</th><th>Comment</th></tr>

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
