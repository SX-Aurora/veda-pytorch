#include "api.h"

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "__ns.h"
//------------------------------------------------------------------------------
static void cumsum_kernel(at::Tensor& result, const at::Tensor& self, int64_t dim) {
	dprint("cumsum_kernel", result, self, dim);
	dim = at::maybe_wrap_dim(dim, self.dim());
	auto result_ = py2veda(result), self_ = py2veda(self);
	CVEDA(veda_tensors_prefix_sum(handle(result), &result_, 0, &self_, dim, 1)); // no carry and always inclusive
}

//------------------------------------------------------------------------------
at::Tensor cumsum(const at::Tensor& self_, int64_t dim, c10::optional<at::ScalarType> dtype)	{
	auto self	= at::native::integer_upcast(self_, dtype);
	auto result	= empty_as(self);
	cumsum_kernel(result, self, dim);
	return result;
}

//------------------------------------------------------------------------------
at::Tensor& cumsum_(at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype) {
	TORCH_CHECK(!dtype.has_value() || (self.scalar_type() == dtype.value()),
		"provided dtype must match the dtype of self tensor in cumsum. Got ",
        at::toString(self.scalar_type()),
        " and ",
        at::toString(dtype.value()),
        ".");
	cumsum_kernel(self, self, dim);
	return self;
}

//------------------------------------------------------------------------------
at::Tensor& cumsum_out(at::Tensor& result, const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype) {
	// result type is favored over dtype; check that they match if provided (NumPy doesn't check)
	TORCH_CHECK(
		!dtype.has_value() || (result.scalar_type() == dtype.value()),
		"provided dtype must match dtype of result in cumsum. Got ",
		at::toString(result.scalar_type()),
		" and ",
		at::toString(dtype.value()),
		".");
	cumsum_kernel(result, self.toType(result.scalar_type()), dim);
	return result;
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("cumsum_out",	TORCH_FN(cumsum_out));
	m.impl("cumsum_",		TORCH_FN(cumsum_));
	m.impl("cumsum",		TORCH_FN(cumsum));
}

//------------------------------------------------------------------------------
#include "__ns.h"