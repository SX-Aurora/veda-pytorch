#include "api.h"

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "__ns.h"
//------------------------------------------------------------------------------
template<typename T>
static at::Tensor& arange_(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step, at::Tensor& out) {
	const auto start_	= start.to<T>();
	const auto step_	= step .to<T>();
	out.resize_({(int64_t)std::ceil((end.to<T>() - start_) / step_)});
	auto out_ = py2veda(out);
	if constexpr(std::is_floating_point<T>::value)	{	CVEDA(veda_tensors_arange_float(handle(out), &out_, start_, step_));	}
	else											{	CVEDA(veda_tensors_arange_int  (handle(out), &out_, start_, step_));	}
	return out;
}

//------------------------------------------------------------------------------
static at::Tensor& arange(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step, at::Tensor& out) {
	dprint("arange", start, end, step, out);
	switch(out.scalar_type()) {
		case c10::ScalarType::Byte:		return arange_<uint8_t>	(start, end, step, out);
		case c10::ScalarType::Char:		return arange_<int8_t>	(start, end, step, out);
		case c10::ScalarType::Short:	return arange_<int16_t>	(start, end, step, out);
		case c10::ScalarType::Int:		return arange_<int32_t>	(start, end, step, out);
		case c10::ScalarType::Long:		return arange_<int64_t>	(start, end, step, out);
		case c10::ScalarType::Float:	return arange_<float>	(start, end, step, out);
		case c10::ScalarType::Double:	return arange_<double>	(start, end, step, out);
		default:
			STHROW("arange is not implemented for dtype: " << out.scalar_type());
	}
	return out;
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("arange.start_out", TORCH_FN(arange));
}

//------------------------------------------------------------------------------
#include "__ns.h"
