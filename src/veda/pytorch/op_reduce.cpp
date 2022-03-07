#include "api.h"

#include "__ns.h"
//------------------------------------------------------------------------------
// Reduce
//------------------------------------------------------------------------------
template<VEDATensors_reduce_op OP>
at::Tensor reduce(const at::Tensor& self) {
	GUARD(self);
	Scalar scalar;
	auto self_ = py2veda(self);
	CVEDA(veda_tensors_reduce_scalar(handle(self), &self_, OP, &scalar));
	return toScalarPyTensor(scalar, self);
}

//------------------------------------------------------------------------------
// ReduceDim
//------------------------------------------------------------------------------
template<VEDATensors_reduce_op OP>
static std::tuple<at::Tensor&, at::Tensor&> reduce_out_kernel(at::Tensor& values, at::Tensor& indices, const at::Tensor& self, int64_t dim, bool keepdim) {
	if(self.is_contiguous() && values.is_contiguous() && indices.is_contiguous()) {
		at::native::_dimreduce_setup(values, self, dim);
    	at::native::_dimreduce_setup(indices, self, dim);
		
		auto values_ = py2veda(values), indices_ = py2veda(indices), self_ = py2veda(self);
		CVEDA(veda_tensors_reduce_dim(handle(values), &values_, &indices_, &self_, OP, dim));
		
		if(!keepdim) {
    		values.squeeze_(dim);
    		indices.squeeze_(dim);
		}
	} else {
		TODO();
	}
	return std::tuple<at::Tensor&, at::Tensor&>{values, indices};
}

//------------------------------------------------------------------------------
template<VEDATensors_reduce_op OP>
static std::tuple<at::Tensor&, at::Tensor&> reduce_out_impl(at::Tensor& values, at::Tensor& indicies, const at::Tensor& self, int64_t dim, bool keepdim) {
	auto result = [&]() {
		at::NoNamesGuard guard;
    	return reduce_out_kernel<OP>(values, indicies, self, dim, keepdim);
	}();
	at::namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
	at::namedinference::propagate_names_for_reduction(indicies, self, dim, keepdim);
	return result;
}

//------------------------------------------------------------------------------
template<VEDATensors_reduce_op OP>
static std::tuple<at::Tensor&, at::Tensor&> reduce_out(at::Tensor& values, at::Tensor& indices, const at::Tensor& self, int64_t dim, bool keepdim) {
	TORCH_CHECK(self.layout() == at::Layout::Strided, "max only supports strided layout, got: ", self.layout());
	TORCH_CHECK(self.device() == values.device(), "expected device ", self.device(), " but got ", values.device(), " for max values output");
	TORCH_CHECK(self.device() == indices.device(), "expected device ", self.device(), " but got ", indices.device(), " for indices output");
	dim = at::maybe_wrap_dim(dim, self.dim());

	if(at::native::_dimreduce_return_trivial_no_ident(values, self, dim, keepdim, "max")) {
    	AT_ASSERT(values.dim() == 0);
		indices.resize_({}).fill_(0);
    	return std::forward_as_tuple(values, indices);
	} else {
		return reduce_out_impl<OP>(values, indices, self, dim, keepdim);
	}
}

//------------------------------------------------------------------------------
template<VEDATensors_reduce_op OP>
static std::tuple<at::Tensor, at::Tensor> reduce_indices(const at::Tensor& self, int64_t dim, bool keepdim) {
	THROWIF(self.is_quantized(), "Quantized tensors not supported");
	at::Tensor indices = at::empty({0}, self.options().dtype(at::kLong));
    at::Tensor values  = at::empty({0}, self.options());
	return reduce_out<OP>(values, indices, self, dim, keepdim);
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("min",		TORCH_FN(reduce			<VEDA_TENSORS_REDUCE_MIN>));
	m.impl("max",		TORCH_FN(reduce			<VEDA_TENSORS_REDUCE_MAX>));
	m.impl("min.dim",	TORCH_FN(reduce_indices	<VEDA_TENSORS_REDUCE_MIN>));
	m.impl("max.dim",	TORCH_FN(reduce_indices	<VEDA_TENSORS_REDUCE_MAX>));
}

//------------------------------------------------------------------------------
#include "__ns.h"