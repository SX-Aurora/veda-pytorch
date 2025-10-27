#include "api.h"

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "__ns.h"
//------------------------------------------------------------------------------
static void binary_kernel(at::TensorIterator iter, const VEDATensors_binary_op op) {
	ASSERT(iter.ntensors() == 3);
	auto &A = iter.tensor(0), &B = iter.tensor(1), &C = iter.tensor(2);
	dprint("binary", A, B, C, op);
	auto A_ = py2veda(A), B_ = py2veda(B), C_ = py2veda(C);
	CVEDA(veda_tensors_binary(handle(A), &A_, &B_, &C_, op));
}

//------------------------------------------------------------------------------
template<VEDATensors_binary_op op>
static at::Tensor& binary_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
	auto iter = at::TensorIterator::comparison_op(result, self, other);
	binary_kernel(iter, op);
	return result;
}

//------------------------------------------------------------------------------
template<VEDATensors_binary_op op>
static at::Tensor binary(const at::Tensor& self, const at::Tensor& other) {
	auto out = empty_as(self, at::kBool);
	binary_out<op>(self, other, out);
	return out;
}

//------------------------------------------------------------------------------
template<VEDATensors_binary_op op>
static at::Tensor& binary_scalar_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) {
	return binary_out<op>(self, wrapped_scalar_tensor(self, other), out);
}

//------------------------------------------------------------------------------
template<VEDATensors_binary_op op>
static at::Tensor binary_scalar(const at::Tensor& self, const at::Scalar& other) {
	return binary<op>(self, wrapped_scalar_tensor(self, other));
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	#define LOGICAL(A, B) m.impl("logical_" A ".out", TORCH_FN(binary_out<B>));

	LOGICAL("and", VEDA_TENSORS_BINARY_AND)
	LOGICAL("or",  VEDA_TENSORS_BINARY_OR)
	LOGICAL("xor", VEDA_TENSORS_BINARY_XOR)

	#define BINARY(A, B)\
		m.impl(A ".Scalar_out",	TORCH_FN(binary_scalar_out<B>));\
		m.impl(A ".Scalar",		TORCH_FN(binary_scalar<B>));\
		m.impl(A ".Tensor_out",	TORCH_FN(binary_out<B>));\
		m.impl(A ".Tensor",		TORCH_FN(binary<B>));

	BINARY("eq", VEDA_TENSORS_BINARY_EQ)
	BINARY("ne", VEDA_TENSORS_BINARY_NE)
	BINARY("lt", VEDA_TENSORS_BINARY_LT)
	BINARY("le", VEDA_TENSORS_BINARY_LE)
	BINARY("gt", VEDA_TENSORS_BINARY_GT)
	BINARY("ge", VEDA_TENSORS_BINARY_GE)
}

//------------------------------------------------------------------------------
#include "__ns.h"
