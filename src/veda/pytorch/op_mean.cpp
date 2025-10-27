#include "api.h"

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "__ns.h"
//------------------------------------------------------------------------------
#if TORCH_VERSION_ >= 11300
using IntArrayRef = at::OptionalIntArrayRef;
#else
using IntArrayRef = at::IntArrayRef;
#endif

//------------------------------------------------------------------------------
template<VEDATensors_reduce_op OP>
static inline void x_stub(at::TensorIterator& iter) {
	ASSERT(iter.ntensors() == 2);
	auto &A = iter.tensor(0), &B = iter.tensor(1);
	auto A_ = py2veda(A), B_ = py2veda(B);

	if(A.numel() == 1) {
		ASSERT(iter.num_reduce_dims() == iter.ndim());
		CVEDA(veda_tensors_reduce(handle(A), &A_, &B_, OP));
	} else {
		ASSERT(iter.num_reduce_dims() == 1);
		auto dim = [&] {
			for(int i = 0; i < iter.ndim(); i++)
				if(A.size(i) != B.size(i))
					return i;
			return -1;
		}();
		ASSERT(dim >= 0 && dim < iter.ndim());
		CVEDA(veda_tensors_reduce_dim(handle(A), &A_, 0, &B_, OP, dim));
	}
}

//------------------------------------------------------------------------------
static void mean_stub(at::TensorIterator& iter) {
	x_stub<VEDA_TENSORS_REDUCE_MEAN>(iter);
}

//------------------------------------------------------------------------------
static void sum_stub(at::TensorIterator& iter) {
	auto &A = iter.tensor(0), &B = iter.tensor(1);
	if(A.numel() == B.numel()) {
		CVEDA(vedaMemcpyDtoDAsync(ptr(A), ptr(B), A.nbytes(), 0));
	} else {
		x_stub<VEDA_TENSORS_REDUCE_SUM>(iter);
	}
}

//------------------------------------------------------------------------------
// Helper Functions
//------------------------------------------------------------------------------
static inline at::ScalarType get_dtype_from_result(at::Tensor& result, c10::optional<at::ScalarType> dtype) {
	TORCH_CHECK(result.defined(), "Cannot create a new tensor inside a reduction op. You likely tried to call an operator with an out argument but the out argument was an undefined tensor.");
	if(dtype.has_value()) {
		return dtype.value();
	} else {
		return result.scalar_type();
	}
}

//------------------------------------------------------------------------------
static inline at::ScalarType get_dtype_from_self(const at::Tensor& self, c10::optional<at::ScalarType> dtype, bool promote_integers) {
	if(dtype.has_value()) {
		return dtype.value();
	}
	at::ScalarType src_type = self.scalar_type();
	if(promote_integers && at::isIntegralType(src_type, /*includeBool=*/true)) {
		return at::kLong;
	}
	return src_type;
}

//------------------------------------------------------------------------------
// Mean
//------------------------------------------------------------------------------
static at::Tensor& mean_IntList_out(const at::Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<c10::ScalarType> opt_dtype, at::Tensor& result) {
	at::ScalarType dtype = get_dtype_from_result(result, opt_dtype);
	auto iter = at::native::make_reduction("mean", result, self, dim, keepdim, dtype);
	mean_stub(iter);
	return result;
}

//------------------------------------------------------------------------------
static at::Tensor mean_dim_IntList(const at::Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> opt_dtype) {
	at::ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
	at::Tensor result = at::native::create_reduction_result(self, dim, keepdim, dtype);
	return mean_IntList_out(self, dim, keepdim, dtype, result);
}

//------------------------------------------------------------------------------
static at::Tensor mean_dim_DimnameList(const at::Tensor& self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
	return mean_dim_IntList(self, at::dimnames_to_positions(self, dim), keepdim, dtype);
}

//------------------------------------------------------------------------------
static at::Tensor& mean_DimnameList_out(const at::Tensor& self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> opt_dtype, at::Tensor& result) {
	return mean_IntList_out(self, at::dimnames_to_positions(self, dim), keepdim, opt_dtype, result);
}

//------------------------------------------------------------------------------
static at::Tensor mean(const at::Tensor& self, c10::optional<c10::ScalarType> dtype) {
	return mean_dim_IntList(self, std::vector<int64_t>{}, false, dtype);
}

//------------------------------------------------------------------------------
// Sum
//------------------------------------------------------------------------------
static at::Tensor& sum_IntList_out(const at::Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<c10::ScalarType> opt_dtype, at::Tensor& result) {
	at::ScalarType dtype = get_dtype_from_result(result, opt_dtype);
	auto iter = at::native::make_reduction("sum", result, self, dim, keepdim, dtype);
	if(iter.numel() == 0)
		result.zero_();
	else
		sum_stub(iter);
	return result;
}

//------------------------------------------------------------------------------
static at::Tensor sum_dim_IntList(const at::Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> opt_dtype) {
	at::ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
	at::Tensor result = at::native::create_reduction_result(self, dim, keepdim, dtype);
	return sum_IntList_out(self, dim, keepdim, dtype, result);
}

//------------------------------------------------------------------------------
static at::Tensor sum_dim_DimnameList(const at::Tensor& self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
	return sum_dim_IntList(self, at::dimnames_to_positions(self, dim), keepdim, dtype);
}

//------------------------------------------------------------------------------
static at::Tensor& sum_DimnameList_out(const at::Tensor& self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> opt_dtype, at::Tensor& result) {
	return sum_IntList_out(self, at::dimnames_to_positions(self, dim), keepdim, opt_dtype, result);
}

//------------------------------------------------------------------------------
static at::Tensor sum(const at::Tensor& self, c10::optional<c10::ScalarType> dtype) {
	return sum_dim_IntList(self, std::vector<int64_t>{}, false, dtype);
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("mean",					TORCH_FN(mean));
	m.impl("mean.dim",				TORCH_FN(mean_dim_IntList));
	m.impl("mean.names_dim",		TORCH_FN(mean_dim_DimnameList));
	m.impl("mean.out",				TORCH_FN(mean_IntList_out));
	m.impl("mean.names_out",		TORCH_FN(mean_DimnameList_out));
	m.impl("sum",					TORCH_FN(sum));
	m.impl("sum.dim_IntList",		TORCH_FN(sum_dim_IntList));
	m.impl("sum.dim_DimnameList",	TORCH_FN(sum_dim_DimnameList));
	m.impl("sum.IntList_out",		TORCH_FN(sum_IntList_out));
	m.impl("sum.DimnameList_out",	TORCH_FN(sum_DimnameList_out));
}

//------------------------------------------------------------------------------
#include "__ns.h"