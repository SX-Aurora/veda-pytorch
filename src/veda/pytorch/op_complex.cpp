#include "api.h"
#include <ATen/native/ComplexHelper.h> // don't move to api.h, as it defines symbols :facepalm:

#include "__ns.h"
//------------------------------------------------------------------------------
static at::Tensor select(const at::Tensor& self, int64_t dim, int64_t index) {
	dprint("select", self, dim, index);

	auto ndim = self.dim();
	dim = at::maybe_wrap_dim(dim, ndim);

	std::vector<int64_t> sizes(self.sizes().begin(), self.sizes().end());
	size_t ocnt		= 1;
	size_t ostride	= 1;
	size_t icnt		= 1;

	for(int64_t i = 0;     i < dim;  i++)	ocnt	*= sizes[i];
	for(int64_t i = dim;   i < ndim; i++)	ostride	*= sizes[i];
	for(int64_t i = dim+1; i < ndim; i++)	icnt	*= sizes[i];

	size_t offset = index * icnt;

	sizes.erase(sizes.begin() + dim);
	
	auto out = empty_as(sizes, self);
	auto out_ = py2veda(out), self_ = py2veda(self);
	CVEDA(veda_tensors_select(handle(out), &out_, &self_, ocnt, ostride, icnt, offset));
	return out;
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("select.int",		TORCH_FN(select));
	m.impl("view_as_real",		TORCH_FN(at::native::view_as_real));
	m.impl("view_as_complex",	TORCH_FN(at::native::view_as_complex));
}

//------------------------------------------------------------------------------
#include "__ns.h"
