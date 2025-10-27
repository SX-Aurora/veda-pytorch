#include "api.h"

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "__ns.h"
//------------------------------------------------------------------------------
static at::Tensor& zero_(at::Tensor& self) {
	dprint("zero_", self);
	CVEDA(vedaMemsetD8Async(ptr(self), 0, self.nbytes(), 0));
	return self;
}

//------------------------------------------------------------------------------
static at::Tensor& fill_(at::Tensor& self, const at::Scalar& value) {
	dprint("fill_", self, value);
	auto s = scalar(self.scalar_type(), value);
	auto self_ = py2veda(self);
	CVEDA(veda_tensors_fill(handle(self), &self_, s));
	return self;
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("zero_",			TORCH_FN(zero_));
	m.impl("fill_.Scalar",	TORCH_FN(fill_));
}

//------------------------------------------------------------------------------
#include "__ns.h"