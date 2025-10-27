#include "api.h"

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "__ns.h"
//------------------------------------------------------------------------------
using DimMask = at::TensorIterator::DimMask;

//------------------------------------------------------------------------------
inline at::Tensor review_reduce_result(const at::Tensor& result, int ndim, DimMask mask, bool keepdim) {
	if(keepdim) {
		return result;
	}
  	auto shape = at::DimVector(result.sizes());
	auto stride = at::DimVector(result.strides());
	for(int dim = 0; dim < ndim; dim++) {
		if(mask[dim]) {
    		shape.insert(shape.begin() + dim, 1);
    		stride.insert(stride.begin() + dim, 0);
		}
	}
	return result.as_strided(shape, stride);
}

//------------------------------------------------------------------------------
inline DimMask make_dim_mask(at::IntArrayRef dims, int64_t ndim) {
	auto mask = DimMask();
	if(dims.empty()) {
		mask.flip();
	} else {
    	for(int64_t dim : dims) {
    		int64_t pos_dim = at::maybe_wrap_dim(dim, ndim);
    		THROWIF(!(pos_dim < 64), "PyTorch doesn't support reduction operations for dim>=64");
      		mask.set(pos_dim);
    	}
  	}
	return mask;
}

//------------------------------------------------------------------------------
inline void allocate_reduction_result(at::Tensor& result, const at::Tensor& self, DimMask mask, bool keepdim, at::ScalarType dtype) {
	auto shape = at::DimVector(self.sizes());
	for(int dim = shape.size() - 1; dim >= 0; dim--) {
		if(mask[dim]) {
      		if(keepdim) {
				shape[dim] = 1;
			} else {
				shape.erase(shape.begin() + dim);
      		}
		}
	}
	if(result.defined()) {
		result.resize_(shape);
	} else {
		result = at::empty(shape, self.options().dtype(dtype));
	}
}

//------------------------------------------------------------------------------
static at::TensorIterator make_reduction(const char* name, at::Tensor& result, const at::Tensor& self, at::IntArrayRef dim, bool keepdim, at::ScalarType in_dtype, at::ScalarType out_dtype) {
	// check that result type and dtype match if provided
	THROWIF(!(!result.defined() || result.scalar_type() == out_dtype), "provided dtype must match dtype of result");
	int64_t ndim = self.dim();
	auto mask = make_dim_mask(dim, ndim);
	allocate_reduction_result(result, self, mask, keepdim, out_dtype);
	auto viewed_result = review_reduce_result(result, ndim, mask, keepdim);
	at::namedinference::propagate_names_for_reduction(result, self, dim, keepdim);
	if(self.scalar_type() == in_dtype) {
		return at::TensorIterator::reduce_op(viewed_result, self);
	}
	return at::TensorIterator::reduce_op(viewed_result, self.to(in_dtype));
}

//------------------------------------------------------------------------------
static at::TensorIterator make_reduction(const char* name, at::Tensor& result, const at::Tensor& self, at::IntArrayRef dim, bool keepdim, at::ScalarType out_dtype) {
	return make_reduction(name, result, self, dim, keepdim, out_dtype, out_dtype);
}

//------------------------------------------------------------------------------
static at::ScalarType get_dtype(at::Tensor& result, const at::Tensor& self, c10::optional<at::ScalarType> dtype, bool promote_integers=false) {
	if(dtype.has_value()) {
		return dtype.value();
	} else if (result.defined()) {
		return result.scalar_type();
	}
  	at::ScalarType src_type = self.scalar_type();
	if(promote_integers && at::isIntegralType(src_type, /*includeBool=*/true)) {
		return at::kLong;
	}
	return src_type;
}

//------------------------------------------------------------------------------
static void norm_kernel(at::TensorIterator& iter, at::Scalar p, at::IntArrayRef dims) {
	float val;
	if		(p.isIntegral(false))	val = (float)	p.to<int64_t>	();
	else if (p.isFloatingPoint())	val = 			p.to<float>		();
	else THROW("norm_kernel_tensor_iterator_impl expects norm to be integer or float");

	auto &A = iter.tensor(0), &B = iter.tensor(1);
	THROWIF(dtype(A) != dtype(B), "torch.norm only supports identical storage types");

	VEDATensors_reduce_op op;
	if		(val == 0.0f)		op = VEDA_TENSORS_REDUCE_L0;
	else if	(val == 1.0f)		op = VEDA_TENSORS_REDUCE_L1;
	else if	(val == 2.0f)		op = VEDA_TENSORS_REDUCE_L2;
	else if	(val == INFINITY)	op = VEDA_TENSORS_REDUCE_MAX;
	else if	(val == -INFINITY)	op = VEDA_TENSORS_REDUCE_MIN;
	else	THROW("Unknown torch.norm type: %f", val);

	auto A_ = py2veda(A), B_ = py2veda(B);
	if(dims.size() == 0) {
		CVEDA(veda_tensors_reduce(handle(A), &A_, &B_, op));
	} else if(dims.size() == 1) {
		CVEDA(veda_tensors_reduce_dim(handle(A), &A_, 0, &B_, op, dims[0]));
	} else {
		THROW("Unsupported torch.norm");
	}
}

//------------------------------------------------------------------------------
static at::Tensor& _norm(const at::Tensor& self, const c10::optional<at::Scalar>& opt_p, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> opt_dtype, at::Tensor& result) {
	auto p = opt_p.value_or(2.0);
	THROWIF(self.layout() != at::Layout::Strided, "norm only supports strided layout");

	at::ScalarType scalarType = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
	THROWIF(!(at::isFloatingType(scalarType) || at::isComplexType(scalarType)),
		"Can only calculate the mean of floating types.");

	at::ScalarType dtype = get_dtype(result, self, opt_dtype, true);
	auto iter = make_reduction("norm", result, self, dim, keepdim, dtype);
	if(iter.numel() == 0)	result.zero_();
	else					norm_kernel(iter, p, dim);
	return result;
}

//------------------------------------------------------------------------------
static at::Tensor& norm_dtype_out(const at::Tensor& self, const c10::optional<at::Scalar>& p, at::IntArrayRef dim, bool keepdim, c10::ScalarType dtype, at::Tensor& result) {
	return _norm(self, p, dim, keepdim, c10::optional<at::ScalarType>(dtype), result);
}

//------------------------------------------------------------------------------
static at::Tensor& norm_out(const at::Tensor& self, const c10::optional<at::Scalar>& p, at::IntArrayRef dim, bool keepdim, at::Tensor& result) {
	return norm_dtype_out(self, p, dim, keepdim, {}, result);
}

//------------------------------------------------------------------------------
inline at::Tensor norm(const at::Tensor& self, const c10::optional<at::Scalar>& p, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> opt_dtype) {
	at::Tensor result;
	return _norm(self, p, dim, keepdim, opt_dtype, result);
}

//------------------------------------------------------------------------------
static at::Tensor norm_ScalarOpt_dim_dtype(const at::Tensor& self, const c10::optional<at::Scalar>& p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) {
	return norm(self, p, dim, keepdim, c10::optional<at::ScalarType>(dtype));
}

//------------------------------------------------------------------------------
static at::Tensor norm_ScalarOpt_dtype(const at::Tensor& self, const c10::optional<at::Scalar>& p, at::ScalarType dtype) {
	return norm(self, p, at::IntArrayRef{}, false, c10::optional<at::ScalarType>(dtype));
}

//------------------------------------------------------------------------------
static at::Tensor norm_ScalarOpt_dim(const at::Tensor& self, const c10::optional<at::Scalar>& p, at::IntArrayRef dim, bool keepdim) {
	return norm(self, p, dim, keepdim, c10::nullopt);
}

//------------------------------------------------------------------------------
static at::Tensor norm_Scalar(const at::Tensor& self, const c10::Scalar& p) {
	THROWIF(self.is_sparse(), "VEDA PyTorch does not support sparse tensors");
	THROWIF(!(self.layout() == at::Layout::Strided), "norm only supports strided layout");
	THROWIF(!(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type())), "norm only supports floating-point dtypes");
	at::Tensor result;
	return _norm(self, p, at::IntArrayRef{}, false, {}, result);
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("norm.ScalarOpt_dim",		TORCH_FN(norm_ScalarOpt_dim));
	m.impl("norm.Scalar",				TORCH_FN(norm_Scalar));
	m.impl("norm.ScalarOpt_dtype",		TORCH_FN(norm_ScalarOpt_dtype));
	m.impl("norm.ScalarOpt_dim_dtype",	TORCH_FN(norm_ScalarOpt_dim_dtype));
	m.impl("norm.dtype_out",			TORCH_FN(norm_dtype_out));
	m.impl("norm.out",					TORCH_FN(norm_out));
}

//------------------------------------------------------------------------------
#include "__ns.h"