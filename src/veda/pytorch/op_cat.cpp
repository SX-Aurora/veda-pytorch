#include "api.h"

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "__ns.h"
//------------------------------------------------------------------------------
static at::Tensor cat(at::TensorList tensors, int64_t dim) {
	// TODO: checks that all dims match up!
	assert(tensors.size() > 0);
	auto& T = tensors.front();
	if(tensors.size() == 1)
		return T;

	GUARD(tensors);

	// Get Sizes ---------------------------------------------------------------
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
	auto output = empty(sizes, c10::typeMetaToScalarType(T.dtype()), T.layout(), T.device(), false, at::MemoryFormat::Contiguous);

	// Call VE -----------------------------------------------------------------
	std::vector<VEDATensors_tensor> inputs;
	inputs.reserve(tensors.size());
	
	for(auto& t : tensors)
		inputs.emplace_back(py2veda(t));

	auto output_ = py2veda(output);
	CVEDA(veda_tensors_cat(handle(output), (int)inputs.size(), inputs.data(), &output_, (int)dim));

	return output;
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("_cat", TORCH_FN(cat));
	// TODO: m.impl("_cat.out", torch::dispatch(DispatchKey::CPU, torch::CppFunction::makeFromUnboxedFunction(&CPUType::_cat_out_out)));
}

//------------------------------------------------------------------------------
#include "__ns.h"