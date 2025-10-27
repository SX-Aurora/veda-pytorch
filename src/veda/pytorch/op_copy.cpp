#include "api.h"

#include "__ns.h"
//------------------------------------------------------------------------------
inline bool isTransposed(const at::Tensor& dst, const at::Tensor& src) {
	return (
		src.dim()			== 2					&&
		dst.is_contiguous()							&&
		dst.scalar_type()	== src.scalar_type()	&&
		dst.device()		== src.device()			&&
		src.size(0)			== dst.size(1)			&&
		src.size(1)			== dst.size(0)			&&
		src.stride(0)		== 1					&&
		src.stride(1)		== src.size(0)
	);
}

//------------------------------------------------------------------------------
static at::Tensor& copy_(at::Tensor& dst, const at::Tensor& src_, bool non_blocking) {
	constexpr int None	= 0;
	constexpr int H2D	= 1;
	constexpr int D2H	= 2;
	constexpr int D2D	= 3;

	dprint("at::copy_", dst, src_, non_blocking);
	
	const size_t bytes = dst.numel() * dst.element_size();
	if(!bytes)
		return dst;

	/**
	 * VEDA requires contiguous tensors. If `src` is not contiguous then we
	 * can't use `non_blocking` as the clone will be destroyed before the memcpy
	 * is complete.
	 */
	at::Tensor src = src_;
	if(!src.is_contiguous()) {
		src = src.clone(at::MemoryFormat::Contiguous);
		non_blocking = false;
	}

	GUARD(src.device().type() == c10::DeviceType::VE ? src.device() : dst.device());

	const auto mode = [&] {
		const auto dst_type	= dst.device().type();
		const auto src_type	= src.device().type();
		if(dst_type == c10::DeviceType::CPU) {
			if(src_type == c10::DeviceType::VE)
				return D2H;
		} else if(dst_type == c10::DeviceType::VE) {
			if(src_type == c10::DeviceType::CPU)	return H2D;
			if(dst_type == c10::DeviceType::VE)		return D2D;
		}
		return None;
	}();
	
	if(mode == None) {
		STHROW("Unable to copy data between " << src.device() << " and " << dst.device() << "!");
	}
	
	const bool isTranspose	= isTransposed(dst, src);
	const bool isConvert	= dst.dtype() != src.dtype();
	if(isTranspose && isConvert) {
		std::ostringstream ss;
		ss << "Can't copy "; dprint_(ss, src); ss << " to "; dprint_(ss, dst);
		THROW(ss.str().c_str());
	}

	auto convert = [](at::Tensor& dst, const at::Tensor& src) -> at::Tensor& {
		auto dst_ = py2veda(dst), src_ = py2veda(src);
		if(isBool(dst))	{	CVEDA(veda_tensors_binary_s	(handle(dst), &dst_, &src_, {}, VEDA_TENSORS_BINARY_NE));	}
		else			{	CVEDA(veda_tensors_convert	(handle(dst), &dst_, &src_));								}
		return dst;
	};

	auto transpose = [](at::Tensor& dst, const at::Tensor& src) -> at::Tensor& {
		auto dst_ = py2veda(dst), src_ = py2veda(src);
		CVEDA(veda_tensors_transpose(handle(dst), &dst_, &src_));
		return dst;
	};

	auto copy = [mode](at::Tensor& dst, const at::Tensor& src, const int line) -> at::Tensor& {
		const size_t bytes		= dst.numel() * dst.element_size();
		const auto dst_bytes	= dst.storage().nbytes();
		const auto src_bytes	= src.storage().nbytes();
		THROWIF(dst_bytes < bytes, "Dst Tensor is too small. Expected %llu but is %llu", (size_t)bytes, (size_t)dst_bytes);
		THROWIF(src_bytes < bytes, "Src Tensor is too small. Expected %llu but is %llu", (size_t)bytes, (size_t)src_bytes);

		auto check = [&](const VEDAresult res) {
			if(res != VEDA_SUCCESS) {
				const char* err;
				vedaGetErrorName(res, &err);
				STHROWAT(__FILE__, line, "Unable to copy " << bytes << "B (" << dst.dtype() << ") from " << src << " (" << src.device() << ") to " << dst << " (" << dst.device() << ")! Caused by: " << err);
			}
		};
		
		ASSERT(dst.is_contiguous());

		if		(mode == H2D)	{	check(vedaMemcpyHtoDAsync((VEDAdeviceptr)dst.data_ptr(), src.data_ptr(), bytes, 0));				}
		else if	(mode == D2H)	{	check(vedaMemcpyDtoHAsync(dst.data_ptr(), (VEDAdeviceptr)src.data_ptr(), bytes, 0));				}
		else					{	check(vedaMemcpyDtoDAsync((VEDAdeviceptr)dst.data_ptr(), (VEDAdeviceptr)src.data_ptr(), bytes, 0));	}

		return dst;
	};

	auto empty_like = [](const at::Tensor& like, const at::Tensor& device) {
		return at::empty_strided(like.sizes(), like.strides(), like.options().device(device.device()));
	};

	if(isConvert) {
		if(mode == H2D) {
			if(src.element_size() == dst.element_size())	{									convert	(dst, copy(dst, src, __LINE__));	}
			else											{	auto x = empty_like(src, dst);	convert	(dst, copy(x,   src, __LINE__));	}
		} else if(mode == D2H) {
																auto x = empty_like(dst, src);	copy	(dst, convert(x, src), __LINE__);	
		} else {
																								convert	(dst, src);
		}
	} else if(isTranspose) {
		if		(mode == H2D)	{	auto x = empty_like(src, dst);	transpose	(dst, copy		(x, src,  __LINE__));	}
		else if	(mode == D2H)	{	auto x = empty_like(dst, src);	copy		(dst, transpose	(x, src), __LINE__);	}
		else					{									transpose	(dst, src);								}
	} else {
		copy(dst, src, __LINE__);
	}

	if(!non_blocking)
		CVEDA(vedaStreamSynchronize(0));
	
	return dst;
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("copy_",										TORCH_FN(copy_));
	m.impl("aten::set_.source_Storage",					TORCH_FN(at::native::set_));
	m.impl("aten::set_.source_Storage_storage_offset",	TORCH_FN(at::native::set_storage_cpu_));
}

//------------------------------------------------------------------------------
#include "__ns.h"