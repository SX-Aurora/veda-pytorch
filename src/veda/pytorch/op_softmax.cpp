#include "api.h"

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "__ns.h"
//------------------------------------------------------------------------------
static at::Tensor& softmax_impl(const at::Tensor& in, const int64_t dim, const bool half_to_float, at::Tensor& out, const VEDATensors_softmax_op op) {
	dprint("softmax_impl", in, dim, half_to_float, out);
	auto o = py2veda(out), x = py2veda(in);
	CVEDA(veda_tensors_softmax(handle(out), &o, &x, dim, op));
	return out;
}

//------------------------------------------------------------------------------
at::Tensor& softmax_out(const at::Tensor& in, const int64_t dim, const bool half_to_float, at::Tensor& out) {
	return softmax_impl(in, dim, half_to_float, out, VEDA_TENSORS_SOFTMAX_SOFTMAX);
}

//------------------------------------------------------------------------------
at::Tensor& log_softmax_out(const at::Tensor& in, const int64_t dim, const bool half_to_float, at::Tensor& out) {
	return softmax_impl(in, dim, half_to_float, out, VEDA_TENSORS_SOFTMAX_LOGSOFTMAX);
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("_softmax.out",				TORCH_FN(softmax_out));
	m.impl("_log_softmax.out",			TORCH_FN(log_softmax_out));
}

//------------------------------------------------------------------------------
#include "__ns.h"