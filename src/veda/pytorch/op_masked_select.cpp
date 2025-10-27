#include "api.h"

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "__ns.h"
//------------------------------------------------------------------------------
static at::Tensor& masked_select_out_out(const at::Tensor& self, const at::Tensor& mask, at::Tensor& out) {
	dprint("masked_select_out_out", self, mask, out);
	auto hnd		= handle(self);
	size_t numel	= 0;
	auto mask_		= py2veda(mask);
	CVEDA(veda_tensors_count(hnd, &mask_, &numel));

	if(numel) {
		out.resize_({(int64_t)numel});
		auto out_ = py2veda(out), self_ = py2veda(self);
		CVEDA(veda_tensors_masked_select(hnd, &out_, &self_, &mask_));
	}
	return out;
}

//------------------------------------------------------------------------------
static at::Tensor masked_select(const at::Tensor& self, const at::Tensor& mask) {
	auto output = empty({0}, c10::typeMetaToScalarType(self.dtype()), self.layout(), self.device(), false, at::MemoryFormat::Contiguous);
	return masked_select_out_out(self, mask, output);
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("masked_select.out",	TORCH_FN(masked_select_out_out));
	m.impl("masked_select",		TORCH_FN(masked_select));
}

//------------------------------------------------------------------------------
#include "__ns.h"