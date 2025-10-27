#include "api.h"

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "__ns.h"
//------------------------------------------------------------------------------
#if TORCH_VERSION_ >= 11300
static at::Tensor& cat(const at::ITensorListRef& tensors, int64_t dim, at::Tensor& out) {
#elif TORCH_VERSION_ >= 11200
static at::Tensor& cat(at::TensorList tensors, int64_t dim, at::Tensor& out) {
#endif
	dprint("cat", out, tensors);
	assert(tensors.size() > 0);
	if(tensors.size() == 1) {
		out = tensors.front();
		return out;
	}

	GUARD(tensors);

	// Get Sizes ---------------------------------------------------------------
	auto& T = tensors.front();

	std::vector<int64_t> sizes;
	sizes.reserve(T.sizes().size());
	for(auto& i : T.sizes())
		sizes.emplace_back(i);

	// Calculate new dimSize ---------------------------------------------------
	int64_t dimSize = 0;
	for(auto& t : tensors)
		dimSize += t.sizes().data()[dim];
	sizes[dim] = dimSize;

	// Create output tensor ----------------------------------------------------
	out.resize_(sizes);

	// Call VE -----------------------------------------------------------------
	std::vector<VEDATensors_tensor> inputs;
	inputs.reserve(tensors.size());

	for(auto& t : tensors)
		inputs.emplace_back(py2veda(t));
	
	auto output_ = py2veda(out);
	CVEDA(veda_tensors_cat(handle(out), (int)inputs.size(), inputs.data(), &output_, (int)dim));

	return out;
}

//------------------------------------------------------------------------------
static at::Tensor cat_(at::TensorList tensors, int64_t dim) {
	auto& T		= tensors.front();
	auto out	= empty({0}, c10::typeMetaToScalarType(T.dtype()), T.layout(), T.device(), false, at::MemoryFormat::Contiguous);
	cat(tensors, dim, out);
	return out;
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
#if TORCH_VERSION_ < 11200
	m.impl("_cat", TORCH_FN(cat));
#else
	m.impl("aten::cat.out", TORCH_FN(cat));
#endif
}

//------------------------------------------------------------------------------
#include "__ns.h"