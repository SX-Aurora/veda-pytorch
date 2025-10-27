#include "api.h"

#include "__ns.h"
//------------------------------------------------------------------------------
// B
//------------------------------------------------------------------------------
static at::Tensor& unary_b_kernel(at::Tensor& out, const at::Tensor& self, const VEDATensors_unary_op op) {
	dprint("unary_b_kernel", out, self);
	auto iter = at::TensorIteratorConfig()
    	.check_all_same_dtype(false)
    	.add_output(out)
    	.add_input(self)
    	.build();
	auto &A = iter.tensor(0), &B = iter.tensor(1);
	auto A_ = py2veda(A), B_ = py2veda(B);
	CVEDA(veda_tensors_unary_b(handle(A), &A_, &B_, op));
	return out;
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor unary_b(const at::Tensor& self) {
	auto out = empty_as(self, c10::ScalarType::Bool);
	return unary_b_kernel(out, self, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor& unary_b_out(const at::Tensor& self, at::Tensor& out) {
	out = toType(out, c10::ScalarType::Bool);
	return unary_b_kernel(out, self, OP);
}

//------------------------------------------------------------------------------
// T
//------------------------------------------------------------------------------
static at::Tensor& unary_t_kernel(at::Tensor& out, const at::Tensor& self, const VEDATensors_unary_op op) {
	dprint("unary_t_kernel", out, self);
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
	dprint("unary_c_kernel", out, self);
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
#if TORCH_VERSION_ < 11200
	auto out = empty_as(self, c10::toValueType(self.scalar_type()));
#else
	auto out = empty_as(self, c10::toRealValueType(self.scalar_type()));
#endif
	return unary_c_kernel(out, self, OP);
}

//------------------------------------------------------------------------------
// TT
//------------------------------------------------------------------------------
static at::Tensor& unary_tt_kernel(at::Tensor& out, const at::Tensor& self, const at::Tensor& other, const VEDATensors_unary_op op) {
	dprint("unary_tt_kernel", out, self, other);
	auto iter = at::TensorIterator::binary_op(out, self, sameType(out, sameDevice(out, other)));
	auto &A = iter.tensor(0), &B = iter.tensor(1), &C = iter.tensor(2);
	auto A_ = py2veda(A), B_ = py2veda(B), C_ = py2veda(C);
	CVEDA(veda_tensors_unary_tt(handle(A), &A_, &B_, &C_, op));
	return out;
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor& unary_tt_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
	return unary_tt_kernel(out, self, other, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor unary_tt(const at::Tensor& self, const at::Tensor& other) {
	auto out = empty_as(self);
	return unary_tt_kernel(out, self, other, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor& unary_tt_(at::Tensor& self, const at::Tensor& other) {
	return unary_tt_kernel(self, self, other, OP);
}

//------------------------------------------------------------------------------
// TS
//------------------------------------------------------------------------------
static at::Tensor& unary_ts_kernel(at::Tensor& out, const at::Tensor& self, const at::Scalar& other, const VEDATensors_unary_op op) {
	dprint("unary_ts_kernel", out, self, other);
	auto iter = at::TensorIterator::binary_op(out, self, sameType(out, sameDevice(out, self)));
	auto &A = iter.tensor(0), &B = iter.tensor(1);
	auto A_ = py2veda(A), B_ = py2veda(B);
	CVEDA(veda_tensors_unary_ts(handle(A), &A_, &B_, scalar(out.scalar_type(), other), op));
	return out;
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor& unary_ts_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) {
	return unary_ts_kernel(out, self, other, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor unary_ts(const at::Tensor& self, const at::Scalar& other) {
	auto out = empty_as(self);
	return unary_ts_kernel(out, self, other, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor& unary_ts_(at::Tensor& self, const at::Scalar& other) {
	return unary_ts_kernel(self, self, other, OP);
}

//------------------------------------------------------------------------------
// TTS
//------------------------------------------------------------------------------
static at::Tensor& unary_tts_kernel(at::Tensor& out, const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, const VEDATensors_unary_op op) {
	dprint("unary_tts_kernel", out, self, other, alpha);
	auto iter = at::TensorIterator::binary_op(out, self, sameType(out, sameDevice(out, other)));
	at::native::alpha_check(iter.dtype(), alpha);
	auto &A	= iter.tensor(0), &B = iter.tensor(1), &C = iter.tensor(2);
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
// TSS
//------------------------------------------------------------------------------
static at::Tensor& unary_tss_kernel(at::Tensor& out, const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& beta, const VEDATensors_unary_op op) {
	auto iter = at::TensorIterator::binary_op(out, self, sameType(out, sameDevice(out, self)));
	auto &A	= iter.tensor(0), &B = iter.tensor(1);
	auto A_	= py2veda(A), B_ = py2veda(B);
	CVEDA(veda_tensors_unary_tss(handle(A), &A_, &B_, scalar(out.scalar_type(), alpha), scalar(out.scalar_type(), beta), op));
	return out;
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor& unary_tss_out(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& beta, at::Tensor& out) {
	return unary_tss_kernel(out, self, alpha, beta, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor unary_tss(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& beta) {
	auto out = empty_as(self);
	return unary_tss_kernel(out, self, alpha, beta, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor& unary_tss_(at::Tensor& self, const at::Scalar& alpha, const at::Scalar& beta) {
	return unary_tss_kernel(self, self, alpha, beta, OP);
}

//------------------------------------------------------------------------------
// TTT
//------------------------------------------------------------------------------
static at::Tensor& unary_ttt_kernel(at::Tensor& out, const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const VEDATensors_unary_op op) {
	auto iter = at::TensorIteratorConfig()
		.add_output(out)
		.add_input(self)
		.add_input(tensor1)
		.add_input(tensor2)
		.build();
	auto &A	= iter.tensor(0), &B = iter.tensor(1), &C = iter.tensor(2), &D = iter.tensor(3);
	auto A_	= py2veda(A), B_ = py2veda(B), C_ = py2veda(C), D_ = py2veda(D);
	CVEDA(veda_tensors_unary_ttt(handle(A), &A_, &B_, &C_, &D_, op));
	return out;
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor& unary_ttt_out(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, at::Tensor& out) {
	return unary_ttt_kernel(out, self, tensor1, tensor2, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor unary_ttt(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2) {
	auto out = empty_as(self);
	return unary_ttt_kernel(out, self, tensor1, tensor2, OP);
}

//------------------------------------------------------------------------------
template<VEDATensors_unary_op OP>
static at::Tensor& unary_ttt_(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2) {
	return unary_ttt_kernel(self, self, tensor1, tensor2, OP);
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
	auto &A	= iter.tensor(0), &B = iter.tensor(1), &C = iter.tensor(2), &D = iter.tensor(3);
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
#if TORCH_VERSION_ < 11200
	return self != self;
#else
	return self.ne(self);
#endif
}

//------------------------------------------------------------------------------
// CLAMP
//------------------------------------------------------------------------------
static at::Tensor& clamp_tss_out(const at::Tensor& self, const c10::optional<at::Scalar>& alpha, const c10::optional<at::Scalar>& beta, at::Tensor& out) {
	if(alpha && beta)	return unary_tss_kernel(out, self, alpha.value(), beta.value(), VEDA_TENSORS_UNARY_CLAMP);
	if(alpha)			return unary_ts_kernel(out, self, alpha.value(), VEDA_TENSORS_UNARY_MAX);
	if(beta)			return unary_ts_kernel(out, self, beta.value(), VEDA_TENSORS_UNARY_MIN);
	out = self;
	return out;
}

//------------------------------------------------------------------------------
static at::Tensor clamp_tss(const at::Tensor& self, const c10::optional<at::Scalar>& alpha, const c10::optional<at::Scalar>& beta) {
	auto out = empty_as(self);
	return clamp_tss_out(self, alpha, beta, out);
}

//------------------------------------------------------------------------------
static at::Tensor& clamp_tss_(at::Tensor& self, const c10::optional<at::Scalar>& alpha, const c10::optional<at::Scalar>& beta) {
	return clamp_tss_out(self, alpha, beta, self);
}

//------------------------------------------------------------------------------
static at::Tensor& clamp_ttt_out(const at::Tensor& self, const c10::optional<at::Tensor>& tensor1, const c10::optional<at::Tensor>& tensor2, at::Tensor& out) {
	if(tensor1 && tensor2)	return unary_ttt_kernel(out, self, tensor1.value(), tensor2.value(), VEDA_TENSORS_UNARY_CLAMP);
	if(tensor1)				return unary_tt_kernel(out, self, tensor1.value(), VEDA_TENSORS_UNARY_MAX);
	if(tensor2)				return unary_tt_kernel(out, self, tensor2.value(), VEDA_TENSORS_UNARY_MIN);
	out = self;
	return out;
}

//------------------------------------------------------------------------------
static at::Tensor clamp_ttt(const at::Tensor& self, const c10::optional<at::Tensor>& tensor1, const c10::optional<at::Tensor>& tensor2) {
	auto out = empty_as(self);
	return clamp_ttt_out(self, tensor1, tensor2, out);
}

//------------------------------------------------------------------------------
static at::Tensor& clamp_ttt_(at::Tensor& self, const c10::optional<at::Tensor>& tensor1, const c10::optional<at::Tensor>& tensor2) {
	return clamp_ttt_out(self, tensor1, tensor2, self);
}

//------------------------------------------------------------------------------
// Register
//------------------------------------------------------------------------------
#if 0
pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
pow.Scalar(Scalar self, Tensor exponent) -> Tensor
#endif

TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("isnan",					TORCH_FN(isnan));
	m.impl("abs",					TORCH_FN(unary_c		<VEDA_TENSORS_UNARY_ABS>));
	m.impl("abs.out",				TORCH_FN(unary_t_out	<VEDA_TENSORS_UNARY_ABS>));

	m.impl("ceil",					TORCH_FN(unary_t		<VEDA_TENSORS_UNARY_CEIL>));
	m.impl("ceil.out",				TORCH_FN(unary_t_out	<VEDA_TENSORS_UNARY_CEIL>));
	m.impl("exp",					TORCH_FN(unary_t		<VEDA_TENSORS_UNARY_EXP>));
	m.impl("exp.out",				TORCH_FN(unary_t_out	<VEDA_TENSORS_UNARY_EXP>));
	m.impl("floor",					TORCH_FN(unary_t		<VEDA_TENSORS_UNARY_FLOOR>));
	m.impl("floor.out",				TORCH_FN(unary_t_out	<VEDA_TENSORS_UNARY_FLOOR>));
	m.impl("log",					TORCH_FN(unary_t		<VEDA_TENSORS_UNARY_LOG>));
	m.impl("log.out",				TORCH_FN(unary_t_out	<VEDA_TENSORS_UNARY_LOG>));
	m.impl("neg",					TORCH_FN(unary_t		<VEDA_TENSORS_UNARY_NEG>));
	m.impl("neg.out",				TORCH_FN(unary_t_out	<VEDA_TENSORS_UNARY_NEG>));
	m.impl("reciprocal",			TORCH_FN(unary_t		<VEDA_TENSORS_UNARY_RECIPROCAL>));
	m.impl("reciprocal.out",		TORCH_FN(unary_t_out	<VEDA_TENSORS_UNARY_RECIPROCAL>));
	m.impl("sqrt",					TORCH_FN(unary_t		<VEDA_TENSORS_UNARY_SQRT>));
	m.impl("sqrt.out",				TORCH_FN(unary_t_out	<VEDA_TENSORS_UNARY_SQRT>));
	m.impl("log1p",					TORCH_FN(unary_t		<VEDA_TENSORS_UNARY_LOG1P>));
	m.impl("log1p.out",				TORCH_FN(unary_t_out	<VEDA_TENSORS_UNARY_LOG1P>));
	m.impl("logical_not",			TORCH_FN(unary_b		<VEDA_TENSORS_UNARY_NOT>));
	m.impl("logical_not.out",		TORCH_FN(unary_b_out	<VEDA_TENSORS_UNARY_NOT>));

	m.impl("clamp",					TORCH_FN(clamp_tss));
	m.impl("clamp_",				TORCH_FN(clamp_tss_));
	m.impl("clamp.Tensor",			TORCH_FN(clamp_ttt));
	m.impl("clamp_.Tensor",			TORCH_FN(clamp_ttt_));
	m.impl("clamp.out",				TORCH_FN(clamp_tss_out));
	m.impl("clamp.Tensor_out",		TORCH_FN(clamp_ttt_out));

	m.impl("clamp_min",				TORCH_FN(unary_ts		<VEDA_TENSORS_UNARY_MAX>));
	m.impl("clamp_min_",			TORCH_FN(unary_ts_		<VEDA_TENSORS_UNARY_MAX>));
	m.impl("clamp_min.Tensor",		TORCH_FN(unary_tt		<VEDA_TENSORS_UNARY_MAX>));
	m.impl("clamp_min_.Tensor",		TORCH_FN(unary_tt_		<VEDA_TENSORS_UNARY_MAX>));
	m.impl("clamp_min.out",			TORCH_FN(unary_ts_out	<VEDA_TENSORS_UNARY_MAX>));
	m.impl("clamp_min.Tensor_out",	TORCH_FN(unary_tt_out	<VEDA_TENSORS_UNARY_MAX>));

	m.impl("clamp_max",				TORCH_FN(unary_ts		<VEDA_TENSORS_UNARY_MIN>));
	m.impl("clamp_max_",			TORCH_FN(unary_ts_		<VEDA_TENSORS_UNARY_MIN>));
	m.impl("clamp_max.Tensor",		TORCH_FN(unary_tt		<VEDA_TENSORS_UNARY_MIN>));
	m.impl("clamp_max_.Tensor",		TORCH_FN(unary_tt_		<VEDA_TENSORS_UNARY_MIN>));
	m.impl("clamp_max.out",			TORCH_FN(unary_ts_out	<VEDA_TENSORS_UNARY_MIN>));
	m.impl("clamp_max.Tensor_out",	TORCH_FN(unary_tt_out	<VEDA_TENSORS_UNARY_MIN>));

	m.impl("mul.Tensor",			TORCH_FN(unary_tt		<VEDA_TENSORS_UNARY_MUL>));
	m.impl("mul.out",				TORCH_FN(unary_tt_out	<VEDA_TENSORS_UNARY_MUL>));
	m.impl("mul_.Tensor",			TORCH_FN(unary_tt_		<VEDA_TENSORS_UNARY_MUL>));	
	m.impl("div.Tensor",			TORCH_FN(unary_tt		<VEDA_TENSORS_UNARY_DIV>));
	m.impl("div.out",				TORCH_FN(unary_tt_out	<VEDA_TENSORS_UNARY_DIV>));
	m.impl("div_.Tensor",			TORCH_FN(unary_tt_		<VEDA_TENSORS_UNARY_DIV>));
	m.impl("maximum",				TORCH_FN(unary_tt		<VEDA_TENSORS_UNARY_MAX>));
	m.impl("maximum.out",			TORCH_FN(unary_tt_out	<VEDA_TENSORS_UNARY_MAX>));
	m.impl("minimum",				TORCH_FN(unary_tt		<VEDA_TENSORS_UNARY_MIN>));
	m.impl("minimum.out",			TORCH_FN(unary_tt_out	<VEDA_TENSORS_UNARY_MIN>));

	m.impl("add.Tensor",			TORCH_FN(unary_tts		<VEDA_TENSORS_UNARY_ADD>));
	m.impl("add.out",				TORCH_FN(unary_tts_out	<VEDA_TENSORS_UNARY_ADD>));
	m.impl("add_.Tensor",			TORCH_FN(unary_tts_		<VEDA_TENSORS_UNARY_ADD>));
	m.impl("sub.Tensor",			TORCH_FN(unary_tts		<VEDA_TENSORS_UNARY_SUB>));
	m.impl("sub.out",				TORCH_FN(unary_tts_out	<VEDA_TENSORS_UNARY_SUB>));
	m.impl("sub_.Tensor",			TORCH_FN(unary_tts_		<VEDA_TENSORS_UNARY_SUB>));

	m.impl("addcdiv.out",			TORCH_FN(unary_ttts_out	<VEDA_TENSORS_UNARY_ADDCDIV>));
	m.impl("addcmul.out",			TORCH_FN(unary_ttts_out	<VEDA_TENSORS_UNARY_ADDCMUL>));

	m.impl("pow.Tensor_Tensor_out",	TORCH_FN(unary_tt_out	<VEDA_TENSORS_UNARY_POW>));
	m.impl("pow.Tensor_Tensor",		TORCH_FN(unary_tt		<VEDA_TENSORS_UNARY_POW>));
	m.impl("pow.Tensor_Scalar_out",	TORCH_FN(unary_ts_out	<VEDA_TENSORS_UNARY_POW>));
	m.impl("pow.Tensor_Scalar",		TORCH_FN(unary_ts		<VEDA_TENSORS_UNARY_POW>));
	m.impl("pow_.Scalar",			TORCH_FN(unary_ts_		<VEDA_TENSORS_UNARY_POW>));
	m.impl("pow_.Tensor",			TORCH_FN(unary_tt_		<VEDA_TENSORS_UNARY_POW>));
}

//------------------------------------------------------------------------------
#include "__ns.h"