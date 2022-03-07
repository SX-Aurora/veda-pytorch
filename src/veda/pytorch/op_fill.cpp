#include "api.h"

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "__ns.h"
//------------------------------------------------------------------------------
static at::Tensor& zero_(at::Tensor& self) {
	GUARD(self);
	CVEDA(vedaMemsetD8Async(ptr(self), 0, self.nbytes(), 0));
	return self;
}

//------------------------------------------------------------------------------
static at::Tensor& fill_(at::Tensor& self, const at::Scalar& value) {
	Scalar s = scalar(self.scalar_type(), value);

	switch(veda_tensors_dtype_bytes(dtype(self))) {
		case 1:		CVEDA(vedaMemsetD8Async		(ptr(self), *(const uint8_t*)&s,	self.numel(),	0));	break;
		case 2:		CVEDA(vedaMemsetD16Async	(ptr(self), *(const uint16_t*)&s,	self.numel(),	0));	break;
		case 4:		CVEDA(vedaMemsetD32Async	(ptr(self), *(const uint32_t*)&s,	self.numel(),	0));	break;
		case 8:		CVEDA(vedaMemsetD64Async	(ptr(self), s.x,					self.numel(),	0));	break;
		case 16:	CVEDA(vedaMemsetD128Async	(ptr(self), s.x, s.y,				self.numel(),	0));	break;
		default:	FAIL();
	}

	return self;
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("zero_",			TORCH_FN(zero_));
	m.impl("fill_.Scalar",	TORCH_FN(fill_));
}

//------------------------------------------------------------------------------
#include "__ns.h"