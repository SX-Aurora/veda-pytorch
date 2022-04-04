#include "api.h"

#include "__ns.h"
//------------------------------------------------------------------------------
inline bool isTransposed(const at::Tensor& self, const at::Tensor& src) {
	return self.is_contiguous() && src.numel() != 0 && src.dim() == 2 &&
		src.stride(0) == 1 && src.stride(1) == src.size(0) &&
		self.scalar_type() == src.scalar_type() && self.device() == src.device();
}

//------------------------------------------------------------------------------
static at::Tensor& copy_(at::Tensor& dst_tensor, const at::Tensor& src_tensor, bool non_blocking) {
	auto dst_device = dst_tensor.device();
	auto src_device = src_tensor.device();

	GUARD(src_device.type() == DEVICE_TYPE ? src_device : dst_device);

	auto dst_type = dst_device.type();
	auto src_type = src_device.type();

	auto dst_dtype = dtype(dst_tensor);
	auto src_dtype = dtype(src_tensor);

	void* dst = dst_tensor.data_ptr();
	void* src = src_tensor.data_ptr();
	
	bool isTranspose	= isTransposed(dst_tensor, src_tensor);
	bool isConvert		= dst_device == src_device && dst_tensor.dtype() != src_tensor.dtype();
	bool isCopy			= !isTranspose && !isConvert && dst_tensor.is_contiguous();

	THROWIF(((int)isTranspose + (int)isConvert + (int)isCopy) != 1, "Unable to determine Copy mode");

	// Convert -----------------------------------------------------------------
	if(isConvert) {
		auto dst_ = py2veda(dst_tensor), src_ = py2veda(src_tensor);
		CVEDA(veda_tensors_convert(handle(dst_tensor), &dst_, &src_));
	}

	// Transpose ---------------------------------------------------------------
	else if(isTranspose) {
		auto dst_ = py2veda(dst_tensor), src_ = py2veda(src_tensor);
		CVEDA(veda_tensors_transpose(handle(dst_tensor), &dst_, &src_));
	}

	// Copy --------------------------------------------------------------------
	else if(isCopy) {
		ASSERT(dst_dtype == src_dtype);
		size_t bytes	= dst_tensor.numel() * dst_tensor.element_size();
		auto dst_bytes	= dst_tensor.storage().nbytes();
		auto src_bytes	= src_tensor.storage().nbytes();
		THROWIF(dst_bytes < bytes, "Dst Tensor is too small. Expected %llu but is %llu", (size_t)bytes, (size_t)dst_bytes);
		THROWIF(src_bytes < bytes, "Src Tensor is too small. Expected %llu but is %llu", (size_t)bytes, (size_t)src_bytes);

		if(bytes) {
			// Copy Host to VE -------------------------------------------------
			if(src_type == at::DeviceType::CPU && dst_type == DEVICE_TYPE) {
				CVEDA(vedaMemcpyHtoDAsync((VEDAdeviceptr)dst, src, bytes, 0));
				if(!non_blocking)
					CVEDA(vedaStreamSynchronize(0));
			}

			// Copy VE to Host -------------------------------------------------
			else if(src_type == DEVICE_TYPE && dst_type == at::DeviceType::CPU) {
				CVEDA(vedaMemcpyDtoHAsync(dst, (VEDAdeviceptr)src, bytes, 0));
				if(!non_blocking)
					CVEDA(vedaStreamSynchronize(0));
			}

			// Copy VE to VE ---------------------------------------------------
			else if(dst_device == src_device) {
				CVEDA(vedaMemcpyDtoDAsync((VEDAdeviceptr)dst, (VEDAdeviceptr)src, bytes, 0));
				if(!non_blocking)
					CVEDA(vedaStreamSynchronize(0));
			}

			// Unsupported ----------------------------------------------------
			else {
				FAIL();
			}
		}
	}
	
	return dst_tensor;
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("copy_",	TORCH_FN(copy_));
}

//------------------------------------------------------------------------------
#include "__ns.h"