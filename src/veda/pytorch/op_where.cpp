#include "api.h"

#include "__ns.h"
//------------------------------------------------------------------------------
static at::Tensor _s_where(const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other) {
	dprint("at::where", condition, self, other);
	TORCH_CHECK(self.dtype() == other.dtype(), "expected scalar type ", self.dtype(), " but found ", other.dtype());
	auto ret = at::empty(self.sizes(), self.options());
	auto iter = at::TensorIteratorConfig()
		.check_all_same_dtype(false)
		.add_output(ret)
		.add_input(condition)
		.add_input(self)
		.add_input(other)
		.build();
	auto &A = iter.tensor(0), &B = iter.tensor(1), &C = iter.tensor(2), &D = iter.tensor(3);
	auto A_ = py2veda(A), B_ = py2veda(B), C_ = py2veda(C), D_ = py2veda(D);
	CVEDA(veda_tensors_where(handle(A), &A_, &B_, &C_, &D_));
	return ret;
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
#if TORCH_VERSION_ < 11200
	m.impl("_s_where", _s_where);
#else
	m.impl("aten::where.self", _s_where);
#endif
}

//------------------------------------------------------------------------------
#include "__ns.h"