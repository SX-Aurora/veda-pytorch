#include "api.h"

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "__ns.h"
//------------------------------------------------------------------------------
inline void kernel(at::TensorIterator& iter, const VEDATensors_bitwise_op op) {
	ASSERT(iter.ntensors() == 3);
	auto &A = iter.tensor(0), &B = iter.tensor(1), &C = iter.tensor(2);
	dprint("bitwise", A, B, C, op);
	auto A_ = py2veda(A), B_ = py2veda(B), C_ = py2veda(C);
	CVEDA(veda_tensors_bitwise(handle(A), &A_, &B_, &C_, op));
}

//------------------------------------------------------------------------------
template<VEDATensors_bitwise_op OP>
static at::Tensor& tensor(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
	auto iter = at::TensorIterator::binary_op(result, self, other);
	kernel(iter, OP);
	return result;
}

//------------------------------------------------------------------------------
template<VEDATensors_bitwise_op OP>
static at::Tensor& scalar(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
	return tensor<OP>(self, wrapped_scalar_tensor(self, other), result);
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("bitwise_and.Tensor_out",	TORCH_FN(tensor<VEDA_TENSORS_BITWISE_AND>));
	m.impl("bitwise_and.Scalar_out",	TORCH_FN(scalar<VEDA_TENSORS_BITWISE_AND>));
	m.impl("bitwise_or.Tensor_out",		TORCH_FN(tensor<VEDA_TENSORS_BITWISE_OR>));
	m.impl("bitwise_or.Scalar_out",		TORCH_FN(scalar<VEDA_TENSORS_BITWISE_OR>));
	m.impl("bitwise_xor.Tensor_out",	TORCH_FN(tensor<VEDA_TENSORS_BITWISE_XOR>));
	m.impl("bitwise_xor.Scalar_out",	TORCH_FN(scalar<VEDA_TENSORS_BITWISE_XOR>));
}

//------------------------------------------------------------------------------
#include "__ns.h"
