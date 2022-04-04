#include "api.h"

#include "__ns.h"
//------------------------------------------------------------------------------
// T
//------------------------------------------------------------------------------
static at::Tensor& unary_t_kernel(at::Tensor& out, const at::Tensor& self, const VEDATensors_unary_op op) {
	auto iter = at::TensorIterator::unary_op(out, self);
	auto& A = iter.tensor(0);
	auto& B = iter.tensor(1);
	auto A_ = py2veda(A), B_ = py2veda(B);
	CVEDA(veda_tensors_unary_t(handle(A), &A_, &B_, op));
	return out;
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor unary_t(const at::Tensor& self) {
	auto out = empty_as(self);
	return unary_t_kernel(out, self, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor& unary_t_out(const at::Tensor& self, at::Tensor& out) {
	return unary_t_kernel(out, self, OP);
}

//------------------------------------------------------------------------------
// C
//------------------------------------------------------------------------------
static at::Tensor& unary_c_kernel(at::Tensor& out, const at::Tensor& self, const VEDATensors_unary_op op) {
	if(self.is_complex()) {
		at::TensorIterator iter;
		iter.build(
			at::TensorIteratorConfig()
      		.set_check_mem_overlap(true)
      		.add_output(out)
      		.add_input(self)
      		.cast_common_dtype_to_outputs(false)
      		.enforce_safe_casting_to_output(false)
      		.check_all_same_dtype(false));
		auto& A = iter.tensor(0);
		auto& B = iter.tensor(1);
		auto A_ = py2veda(A), B_ = py2veda(B);
		CVEDA(veda_tensors_unary_c(handle(A), &A_, &B_, op));
		return out;
	}
	return unary_t_kernel(out, self, op);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor unary_c(const at::Tensor& self) {
	auto out = empty_as(self, c10::toValueType(self.scalar_type()));
	return unary_c_kernel(out, self, OP);
}

//------------------------------------------------------------------------------
// TT
//------------------------------------------------------------------------------
static at::Tensor& unary_tt_kernel(at::Tensor& out, const at::Tensor& self, const at::Tensor& other, const VEDATensors_unary_op op) {
	auto iter = at::TensorIterator::binary_op(out, self, sameType(out, sameDevice(out, other)));
	auto A = iter.tensor(0), B = iter.tensor(1), C = iter.tensor(2);
	auto A_ = py2veda(A), B_ = py2veda(B), C_ = py2veda(C);
	CVEDA(veda_tensors_unary_tt(handle(A), &A_, &B_, &C_, op));
	return out;
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
at::Tensor& unary_tt_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
	return unary_tt_kernel(out, self, other, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
at::Tensor unary_tt(const at::Tensor& self, const at::Tensor& other) {
	auto out = empty_as(self);
	return unary_tt_kernel(out, self, other, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
at::Tensor& unary_tt_(at::Tensor& self, const at::Tensor& other) {
	return unary_tt_kernel(self, self, other, OP);
}

//------------------------------------------------------------------------------
// TTS
//------------------------------------------------------------------------------
static at::Tensor& unary_tts_kernel(at::Tensor& out, const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, const VEDATensors_unary_op op) {
	auto iter = at::TensorIterator::binary_op(out, self, sameType(out, sameDevice(out, other)));
	at::native::alpha_check(iter.dtype(), alpha);
	auto A	= iter.tensor(0), B = iter.tensor(1), C = iter.tensor(2);
	auto A_	= py2veda(A), B_ = py2veda(B), C_ = py2veda(C);
	CVEDA(veda_tensors_unary_tts(handle(A), &A_, &B_, &C_, scalar(out.scalar_type(), alpha), op));
	return out;
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor& unary_tts_out(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out) {
	return unary_tts_kernel(out, self, other, alpha, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor unary_tts(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
	auto out = empty_as(self);
	return unary_tts_kernel(out, self, other, alpha, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor& unary_tts_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
	return unary_tts_kernel(self, self, other, alpha, OP);
}

//------------------------------------------------------------------------------
// TTTS
//------------------------------------------------------------------------------
static at::Tensor& unary_ttts_kernel(at::Tensor& result, const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value, const VEDATensors_unary_op op) {
	auto iter = at::TensorIteratorConfig()
		.add_output(result)
		.add_input(self)
		.add_input(tensor1)
		.add_input(tensor2)
		.build();
	auto A	= iter.tensor(0), B = iter.tensor(1), C = iter.tensor(2), D = iter.tensor(3);
	auto A_	= py2veda(A), B_ = py2veda(B), C_ = py2veda(C), D_ = py2veda(D);
	CVEDA(veda_tensors_unary_ttts(handle(A), &A_, &B_, &C_, &D_, scalar(result.scalar_type(), value), op));
	return result;
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor unary_ttts(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value) {
	at::Tensor result = at::empty({0}, self.options());
	return unary_ttts_kernel(result, self, tensor1, tensor2, value, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor& unary_ttts_(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value) {
	return unary_ttts_kernel(self, self, tensor1, tensor2, value, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
at::Tensor& unary_ttts_out(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value, at::Tensor& result) {
	return unary_ttts_kernel(result, self, tensor1, tensor2, value, OP);
}

//------------------------------------------------------------------------------
// ISNAN
//------------------------------------------------------------------------------
at::Tensor isnan(const at::Tensor& self) {
	return self != self;
}

//------------------------------------------------------------------------------
// Register
//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("abs",				TORCH_FN(unary_c		<VEDA_TENSORS_UNARY_ABS>));
	m.impl("abs.out",			TORCH_FN(unary_t_out	<VEDA_TENSORS_UNARY_ABS>));

	m.impl("ceil",				TORCH_FN(unary_t		<VEDA_TENSORS_UNARY_CEIL>));
	m.impl("ceil.out",			TORCH_FN(unary_t_out	<VEDA_TENSORS_UNARY_CEIL>));
	m.impl("floor",				TORCH_FN(unary_t		<VEDA_TENSORS_UNARY_FLOOR>));
	m.impl("floor.out",			TORCH_FN(unary_t_out	<VEDA_TENSORS_UNARY_FLOOR>));
	m.impl("reciprocal",		TORCH_FN(unary_t		<VEDA_TENSORS_UNARY_RECIPROCAL>));
	m.impl("reciprocal.out",	TORCH_FN(unary_t_out	<VEDA_TENSORS_UNARY_RECIPROCAL>));
	m.impl("sqrt",				TORCH_FN(unary_t		<VEDA_TENSORS_UNARY_SQRT>));
	m.impl("sqrt.out",			TORCH_FN(unary_t_out	<VEDA_TENSORS_UNARY_SQRT>));

	m.impl("mul.Tensor",		TORCH_FN(unary_tt		<VEDA_TENSORS_UNARY_MUL>));
	m.impl("mul_.Tensor",		TORCH_FN(unary_tt_		<VEDA_TENSORS_UNARY_MUL>));
	m.impl("mul.out",			TORCH_FN(&unary_tt_out	<VEDA_TENSORS_UNARY_MUL>));
	
	m.impl("div.Tensor",		TORCH_FN(unary_tt		<VEDA_TENSORS_UNARY_DIV>));
	m.impl("div_.Tensor",		TORCH_FN(unary_tt_		<VEDA_TENSORS_UNARY_DIV>));
	m.impl("div.out",			TORCH_FN(unary_tt_out	<VEDA_TENSORS_UNARY_DIV>));

	m.impl("maximum",			TORCH_FN(unary_tt		<VEDA_TENSORS_UNARY_MAX>));
	m.impl("maximum.out",		TORCH_FN(unary_tt_out	<VEDA_TENSORS_UNARY_MAX>));

	m.impl("minimum",			TORCH_FN(unary_tt		<VEDA_TENSORS_UNARY_MIN>));
	m.impl("minimum.out",		TORCH_FN(unary_tt_out	<VEDA_TENSORS_UNARY_MIN>));

	m.impl("isnan",				TORCH_FN(isnan));

	m.impl("add.Tensor",		TORCH_FN(unary_tts		<VEDA_TENSORS_UNARY_ADD>));
	m.impl("add_.Tensor",		TORCH_FN(unary_tts_		<VEDA_TENSORS_UNARY_ADD>));
	m.impl("add.out",			TORCH_FN(unary_tts_out	<VEDA_TENSORS_UNARY_ADD>));

	m.impl("sub.Tensor",		TORCH_FN(unary_tts		<VEDA_TENSORS_UNARY_SUB>));
	m.impl("sub_.Tensor",		TORCH_FN(unary_tts_		<VEDA_TENSORS_UNARY_SUB>));
	m.impl("sub.out",			TORCH_FN(unary_tts_out	<VEDA_TENSORS_UNARY_SUB>));

	m.impl("addcmul.out",		TORCH_FN(unary_ttts_out	<VEDA_TENSORS_UNARY_ADDCMUL>));
	m.impl("addcdiv.out",		TORCH_FN(unary_ttts_out	<VEDA_TENSORS_UNARY_ADDCDIV>));
}

//------------------------------------------------------------------------------
#include "__ns.h"