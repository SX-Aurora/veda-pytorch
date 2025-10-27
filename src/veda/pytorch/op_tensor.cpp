#include "api.h"

#if TORCH_VERSION_ >= 11200
namespace at {
	namespace native {
		at::Tensor	as_strided_tensorimpl	(const Tensor&, IntArrayRef, IntArrayRef, optional<int64_t>);
		at::Tensor	squeeze					(const at::Tensor&);
		at::Tensor	squeeze					(const at::Tensor&, int64_t);
		at::Tensor	unsqueeze				(const at::Tensor&, int64_t);
		at::Tensor	view					(const at::Tensor&, at::IntArrayRef);
		at::Tensor&	squeeze_				(at::Tensor&);
		at::Tensor&	squeeze_				(at::Tensor&, int64_t);
		at::Tensor&	unsqueeze_				(at::Tensor&, int64_t);
	}
}
#endif

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "__ns.h"
//------------------------------------------------------------------------------
static at::Tensor clone(const at::Tensor& self, c10::optional<c10::MemoryFormat> optional_memory_format) {
	dprint("clone", self);
	auto out = empty_as(self);
	CVEDA(vedaMemcpyDtoDAsync(ptr(out), ptr(self), self.nbytes(), 0));
	return out;
}

//------------------------------------------------------------------------------
static at::Scalar _local_scalar_dense(const at::Tensor& self) {
	dprint("_local_scalar_dense", self);
	GUARD(self);
	VEDATensors_scalar value = {};
	CVEDA(vedaMemcpyDtoH(&value, ptr(self), veda_tensors_dtype_bytes(dtype(self))));
	return toPyScalar(self.scalar_type(), value);
}

//------------------------------------------------------------------------------
static const at::Tensor& resize(const at::Tensor& self, at::IntArrayRef sizes, c10::optional<at::MemoryFormat> optional_memory_format) {
	dprint("resize", self, sizes);
	GUARD(self);
	resizePyTensor(self.unsafeGetTensorImpl(), sizes, {});
	if(optional_memory_format.has_value()) {
		auto memory_format = optional_memory_format.value();
		THROWIF(memory_format != at::MemoryFormat::Preserve && memory_format != at::MemoryFormat::Contiguous, "[VE] Unsupported memory format");
	}
	return self;
}

//------------------------------------------------------------------------------
static inline void checkInBoundsForStorage(c10::IntArrayRef size, c10::IntArrayRef stride, int64_t storage_offset, const caffe2::TypeMeta data_type, const c10::Storage& new_storage) {
	int64_t storage_size_bytes = at::detail::computeStorageNbytes(size, stride, data_type.itemsize());
	int64_t storage_offset_bytes = storage_offset * data_type.itemsize();
	if(storage_size_bytes == 0) {
		// NB: (a tensor with arbitrary 0 dims)'s storage can have any numel.
		return;
	}
	int64_t new_storage_size_bytes = new_storage.nbytes();
	TORCH_CHECK(
		storage_size_bytes + storage_offset_bytes <= new_storage_size_bytes,
		"setStorage: sizes ",
		size,
		", strides ",
		stride,
		","
		" storage offset ",
		storage_offset,
		", and itemsize ",
		data_type.itemsize(),
		" requiring a storage size of ",
		storage_size_bytes + storage_offset_bytes,
		" are out of bounds for storage of size ",
		new_storage_size_bytes);
}

//------------------------------------------------------------------------------
inline void setStrided(const at::Tensor& self, c10::IntArrayRef size, c10::IntArrayRef stride, int64_t storage_offset) {
	TORCH_CHECK(size.size() == stride.size(), "mismatch in length of strides and shape");
	auto* self_ = self.unsafeGetTensorImpl();
	checkInBoundsForStorage(size, stride, storage_offset, self_->dtype(), self_->storage());

	/* storage offset */
	TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
	self_->set_storage_offset(storage_offset);

	/* size and stride */
	if(self_->sizes() == size && self_->strides() == stride) {
		return;
	}

	for(auto val : stride) {
		TORCH_CHECK(val >= 0,
			"as_strided: Negative strides are not supported at the moment, "
			"got strides: ", stride);
	}
	self_->set_sizes_and_strides(size, stride);
}

//------------------------------------------------------------------------------
template<typename Vec>
static at::Tensor alias_with_sizes_and_strides(const at::Tensor& self, const Vec& sizes, const Vec& strides) {
	at::Tensor self_ = at::detail::make_tensor<at::TensorImpl>(c10::TensorImpl::VIEW, at::Storage(self.storage()), self.key_set(), self.dtype());
	setStrided(self_, sizes, strides, self.storage_offset());
	at::namedinference::propagate_names(self_, self);
	return self_;
}

//------------------------------------------------------------------------------
at::Tensor _reshape_alias(const at::Tensor& self, c10::IntArrayRef sizes, c10::IntArrayRef strides) {
	return alias_with_sizes_and_strides(self, sizes, strides);
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, DEVICE_TYPE_, m) {
	m.impl("empty.memory_format",	empty);
	m.impl("empty_strided",			TORCH_FN(empty_strided));
	m.impl("resize_",				TORCH_FN(resize));
	m.impl("_local_scalar_dense",	TORCH_FN(_local_scalar_dense));
	m.impl("_reshape_alias",		TORCH_FN(_reshape_alias));
	m.impl("clone",					TORCH_FN(clone));
	m.impl("view", 					TORCH_FN(at::native::view));
	m.impl("as_strided",			TORCH_FN(at::native::as_strided_tensorimpl));
	m.impl("squeeze",				TORCH_FN(static_cast<at::Tensor(*)(const at::Tensor&)>			(&at::native::squeeze)));
	m.impl("squeeze.dim",			TORCH_FN(static_cast<at::Tensor(*)(const at::Tensor&, int64_t)>	(&at::native::squeeze)));
	m.impl("squeeze_",				TORCH_FN(static_cast<at::Tensor&(*)(at::Tensor&)>				(&at::native::squeeze_)));
	m.impl("squeeze_.dim",			TORCH_FN(static_cast<at::Tensor&(*)(at::Tensor&, int64_t)>		(&at::native::squeeze_)));
	m.impl("unsqueeze",				TORCH_FN(static_cast<at::Tensor(*)(const at::Tensor&, int64_t)>	(&at::native::unsqueeze)));
	m.impl("unsqueeze_",			TORCH_FN(static_cast<at::Tensor&(*)(at::Tensor&, int64_t)>		(&at::native::unsqueeze_)));
}

//------------------------------------------------------------------------------
#include "__ns.h"
