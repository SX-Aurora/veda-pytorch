#include "api.h"

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "__ns.h"
//------------------------------------------------------------------------------
static at::Tensor& masked_fill_scalar(at::Tensor& self, const at::Tensor& mask, const at::Scalar& source) {
	dprint("masked_fill_scalar", self, mask, source);
	auto self_ = py2veda(self), mask_ = py2veda(mask);
	CVEDA(veda_tensors_masked_fill(handle(self), &self_, scalar(self.scalar_type(), source), &mask_));
	return self;
}

//------------------------------------------------------------------------------
static at::Tensor& masked_fill_tensor(at::Tensor& self, const at::Tensor& mask, const at::Tensor& source__) {
	dprint("masked_fill_tensor", self, mask, source__);
	auto source = source__.toType(self.scalar_type());
	auto self_ = py2veda(self), mask_ = py2veda(mask), source_ = py2veda(source);
	CVEDA(veda_tensors_masked_fill_t(handle(self), &self_, &source_, &mask_));
	return self;
}

//------------------------------------------------------------------------------
static at::Tensor& masked_scatter(at::Tensor& self, const at::Tensor& mask, const at::Tensor& source__) {
	dprint("masked_scatter", self, mask, source__);
	auto source = source__.toType(self.scalar_type());
	auto self_ = py2veda(self), mask_ = py2veda(mask), source_ = py2veda(source);
	CVEDA(veda_tensors_masked_scatter(handle(self), &self_, &source_, &mask_));
	return self;
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("masked_fill_.Scalar",	TORCH_FN(masked_fill_scalar));
	m.impl("masked_fill_.Tensor",	TORCH_FN(masked_fill_tensor));
	m.impl("masked_scatter_",		TORCH_FN(masked_scatter));
}

//------------------------------------------------------------------------------
#include "__ns.h"
