#include "api.h"

#include "__ns.h"
//------------------------------------------------------------------------------
// Static
//------------------------------------------------------------------------------
static void veFree(void* vptr) {
	// MemFree automatically detects the correct device!
	CVEDA(vedaMemFreeAsync((VEDAdeviceptr)vptr, 0));
}

//------------------------------------------------------------------------------
inline at::StorageImpl* createNewStorage(c10::TensorImpl* self) {
	return c10::make_intrusive<at::StorageImpl>(at::StorageImpl::use_byte_size_t(), 0, allocator(), true).release();
}

//------------------------------------------------------------------------------
inline at::StorageImpl* getStorage(c10::TensorImpl* self) {
	return self->storage().unsafeGetStorageImpl();
}

//------------------------------------------------------------------------------
inline void setStorage(c10::TensorImpl* self, at::StorageImpl* storage) {
	self->set_storage_keep_dtype(at::Storage(c10::intrusive_ptr<at::StorageImpl>::reclaim(storage)));
}

//------------------------------------------------------------------------------
inline at::StorageImpl* addNewStorage(c10::TensorImpl* self) {
	auto storage = createNewStorage(self);
	assert(self->device() == storage->device());
	setStorage(self, storage);
	return storage;
}

//------------------------------------------------------------------------------
// External
//------------------------------------------------------------------------------
VEDATensors_reduce_op reduction(int64_t reduction) { // TODO: move to other file!
	switch(reduction) {
		case at::Reduction::None:	return VEDA_TENSORS_REDUCE_UNKNOWN;
		case at::Reduction::Mean:	return VEDA_TENSORS_REDUCE_MEAN;
		case at::Reduction::Sum:	return VEDA_TENSORS_REDUCE_SUM;
	}
	FAIL();
}

//------------------------------------------------------------------------------
VEDATensors_scalar scalar(const c10::ScalarType& type, const c10::Scalar& value) {
	VEDATensors_scalar scalar = {};
	
	switch(type) {
		case c10::ScalarType::Byte:				scalar.U8		= value.to<uint8_t>	();	return scalar;
		case c10::ScalarType::Char:				scalar.S8		= value.to<int8_t>	();	return scalar;
		case c10::ScalarType::Short:			scalar.S16		= value.to<int16_t>	();	return scalar;
		case c10::ScalarType::Int:				scalar.S32		= value.to<int32_t>	();	return scalar;
		case c10::ScalarType::Long:				scalar.S64		= value.to<int64_t>	();	return scalar;
		case c10::ScalarType::Float:			scalar.F32		= value.to<float>	();	return scalar;
		case c10::ScalarType::Double:			scalar.F64		= value.to<double>	();	return scalar;
		case c10::ScalarType::Bool:				scalar.S8		= value.to<bool>	();	return scalar;
		case c10::ScalarType::ComplexFloat:		{	auto v = value.to<c10::complex<float>>();	memcpy(&scalar, &v, sizeof(float)  * 2);	return scalar;	}
		case c10::ScalarType::ComplexDouble:	{	auto v = value.to<c10::complex<double>>();	memcpy(&scalar, &v, sizeof(double) * 2);	return scalar;	}
	}

	THROW("Unknown scalar type");
}

//------------------------------------------------------------------------------
at::Scalar toPyScalar(const c10::ScalarType& type, const VEDATensors_scalar value) {
	switch(type) {
		case c10::ScalarType::Byte:				return at::Scalar(value.U8);
		case c10::ScalarType::Char:				return at::Scalar(value.S8);
		case c10::ScalarType::Short:			return at::Scalar(value.S16);
		case c10::ScalarType::Int:				return at::Scalar(value.S32);
		case c10::ScalarType::Long:				return at::Scalar(value.S64);
		case c10::ScalarType::Float:			return at::Scalar(value.F32);
		case c10::ScalarType::Double:			return at::Scalar(value.F64);
		case c10::ScalarType::Bool:				return at::Scalar(value.S8);
		case c10::ScalarType::ComplexFloat:		return at::Scalar(*(const c10::complex<float>*)	&value);
		case c10::ScalarType::ComplexDouble:	return at::Scalar(*(const c10::complex<double>*)&value);
	}

	THROW("Unknown scalar type");
}

//------------------------------------------------------------------------------
at::Tensor empty(at::IntArrayRef size, c10::optional<c10::ScalarType> dtype, c10::optional<c10::Layout> layout, c10::optional<c10::Device> device, c10::optional<bool> pin_memory, const c10::optional<c10::MemoryFormat> memory_format) {
	FAILIF(device->type() != DEVICE_TYPE);
	for(auto x : size)
		THROWIF(x < 0, "Cannot allocate Tensor with negative size!");
	THROWIF(pin_memory && *pin_memory, "NEC SX-Aurora does not support pinned memory!");

	GUARD(*device);

	int64_t nelements = 0;
	nelements = 1;
	for(auto x : size)
		nelements *= x;

	auto alloc		= allocator();
	size_t nbytes	= nelements * c10::elementSize(*dtype);

	auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
		c10::StorageImpl::use_byte_size_t(),
		nbytes,
		alloc->allocate(nbytes),
		alloc,
		true);

	auto tensor = at::detail::make_tensor<c10::TensorImpl>(std::move(storage_impl), DISPATCH_KEY, c10::scalarTypeToTypeMeta(*dtype));
	// Default TensorImpl has size [0]
	if(size.size() != 1 || size[0] != 0) {
		tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
	}

	tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format ? *memory_format : at::MemoryFormat::Contiguous);
	return tensor;
}

//------------------------------------------------------------------------------
size_t numel(const at::Tensor& self) {
	if([&] {
		bool isAnyZero = false;
		for(auto& it : self.strides()) {
			if(it == 0) {
				isAnyZero = true;
			} else if(isAnyZero) {
				STHROW("Detected zero/non-zero mixed stride: " << self.strides());
			}
		}
		return isAnyZero;
	}())
		return 1;
	return self.numel();
}

//------------------------------------------------------------------------------
bool isBool(const at::Tensor& self) {
	return self.scalar_type() == c10::ScalarType::Bool;
}

//------------------------------------------------------------------------------
VEDATensors_dtype dtype(const at::Tensor& self) {
	switch(self.scalar_type()) {
		case c10::ScalarType::Byte:				return VEDA_TENSORS_DTYPE_U8;
		case c10::ScalarType::Char:				return VEDA_TENSORS_DTYPE_S8;
		case c10::ScalarType::Short:			return VEDA_TENSORS_DTYPE_S16;
		case c10::ScalarType::Int:				return VEDA_TENSORS_DTYPE_S32;
		case c10::ScalarType::Long:				return VEDA_TENSORS_DTYPE_S64;
		case c10::ScalarType::Float:			return VEDA_TENSORS_DTYPE_F32;
		case c10::ScalarType::Double:			return VEDA_TENSORS_DTYPE_F64;
		case c10::ScalarType::Bool:				return VEDA_TENSORS_DTYPE_S8;
		case c10::ScalarType::ComplexFloat:		return VEDA_TENSORS_DTYPE_F32_F32;
		case c10::ScalarType::ComplexDouble:	return VEDA_TENSORS_DTYPE_F64_F64;
	}
	
	STHROW("Unknown PyTorch c10::ScalarType: " << self.scalar_type());
}

//------------------------------------------------------------------------------
bool isBool(const c10::TensorImpl* self) {
	return self->dtype() == caffe2::TypeMeta::Make<bool>();
}

//------------------------------------------------------------------------------
VEDATensors_dtype dtype(const c10::TensorImpl* self) {
	if(self->dtype() == caffe2::TypeMeta::Make<bool>())					return VEDA_TENSORS_DTYPE_S8;
	if(self->dtype() == caffe2::TypeMeta::Make<int8_t>())				return VEDA_TENSORS_DTYPE_S8;
	if(self->dtype() == caffe2::TypeMeta::Make<int16_t>())				return VEDA_TENSORS_DTYPE_S16;
	if(self->dtype() == caffe2::TypeMeta::Make<int32_t>())				return VEDA_TENSORS_DTYPE_S32;
	if(self->dtype() == caffe2::TypeMeta::Make<int64_t>())				return VEDA_TENSORS_DTYPE_S64;
	if(self->dtype() == caffe2::TypeMeta::Make<uint8_t>())				return VEDA_TENSORS_DTYPE_U8;
	if(self->dtype() == caffe2::TypeMeta::Make<uint16_t>())				return VEDA_TENSORS_DTYPE_U16;
	if(self->dtype() == caffe2::TypeMeta::Make<float>())				return VEDA_TENSORS_DTYPE_F32;
	if(self->dtype() == caffe2::TypeMeta::Make<double>())				return VEDA_TENSORS_DTYPE_F64;
	if(self->dtype() == caffe2::TypeMeta::Make<c10::complex<float>>())	return VEDA_TENSORS_DTYPE_F32_F32;
	if(self->dtype() == caffe2::TypeMeta::Make<c10::complex<double>>())	return VEDA_TENSORS_DTYPE_F64_F64;
	
	STHROW("Unknown PyTorch caffee2::TypeMeta: " << self->dtype());
}

//------------------------------------------------------------------------------
c10::TensorImpl* resizePyTensor(c10::TensorImpl* self, at::IntArrayRef size, c10::optional<at::IntArrayRef> stride) {
	if(self->sizes() == size && (!stride || self->strides() == stride))
		return self;
	
	GUARD(self);

	int64_t storage_size = 1;
	if(stride) {
		self->set_sizes_and_strides(size, *stride);
		for(size_t dim = 0; dim < size.size(); ++dim) {
			if(size[dim] == 0) {
				storage_size = 0;
				break;
			}
			storage_size += (size[dim] - 1) * stride.value()[dim];
		}
	} else {
		self->set_sizes_contiguous(size);
		storage_size = self->numel();
	}

	storage_size *= self->dtype().itemsize(); 

	if(storage_size > 0) {
		at::StorageImpl* storage = getStorage(self);

		// create new storage if necessary
		if(!storage)
			storage = addNewStorage(self);
		
		// does the storage needs to be resized?
		auto newSize = storage_size + self->storage_offset();
		if(newSize > storage->nbytes()) {
			assert(newSize > 0);

			if(storage->resizable()) {
				at::DataPtr newData(storage->allocator()->allocate(newSize));
				at::DataPtr oldData	= storage->set_data_ptr(std::move(newData));
				size_t oldSize		= storage->nbytes();
				storage->set_nbytes(newSize);

				if(oldSize) {
					if(auto copySize = std::max(storage->nbytes(), oldSize))
						CVEDA(vedaMemcpyDtoDAsync((VEDAdeviceptr)storage->data(), (VEDAdeviceptr)oldData.get(), copySize, 0));
				}
			} else {
				THROW("[VE] unresizeable storage?!?");
			}
		}
	}

	return self;
}

//------------------------------------------------------------------------------
at::Allocator* allocator(void) {
	class VEAllocator final : public at::Allocator {
		virtual at::DataPtr allocate(size_t nbytes)
		#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
		#else
			const
		#endif
		override {
			auto device = getGuardImpl()->getDevice();
			GUARD(device);
			VEDAdeviceptr ptr = 0;
			if(nbytes)
				CVEDA(vedaMemAllocAsync(&ptr, nbytes, 0));
			return {ptr, ptr, &veFree, device};
		};

		virtual at::DeleterFnPtr raw_deleter(void) const override {
			return &veFree;
		}

		#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
		virtual void copy_data(void* dest, const void* src, std::size_t count) const override {
			CVEDA(vedaMemcpyDtoDAsync((VEDAdeviceptr)dest, (VEDAdeviceptr)src, count, 0));
		}
		#endif
	};
	static VEAllocator s_allocator;
	return &s_allocator;
}

//------------------------------------------------------------------------------
VEDATensors_handle handle(void) {
	VEDAcontext ctx;
	VEDATensors_handle handle;
	CVEDA(vedaCtxGetCurrent(&ctx));
	CVEDA(veda_tensors_get_handle_by_ctx(&handle, ctx));
	return handle;
}

//------------------------------------------------------------------------------
VEDATensors_handle handle(const at::Tensor& self) {
	ASSERT(self.device().index() >= 0);
	VEDATensors_handle handle;
	CVEDA(veda_tensors_get_handle_by_id(&handle, self.device().index()));
	return handle;
}

//------------------------------------------------------------------------------
at::Tensor sameDevice(const at::Tensor& self, at::Tensor other) {
	return self.device() != other.device() ? other.to(self.device()) : other;
}

//------------------------------------------------------------------------------
at::Tensor toType(at::Tensor tensor, const c10::ScalarType dtype) {
	return tensor.scalar_type() != dtype ? tensor.toType(dtype) : tensor;
}

//------------------------------------------------------------------------------
at::Tensor sameType(const at::Tensor& self, at::Tensor other) {
	return toType(other, self.scalar_type());
}

//------------------------------------------------------------------------------
VEDATensors_tensor py2veda(const at::Tensor& self) {
	auto sizes = self.sizes();
	if([&] {
		bool allZero = false;
		auto strides = self.strides();
		
		for(size_t i = 0; i < sizes.size(); i++) {
			if(sizes[i] > 1) {
				if(strides[i] == 0) {
					allZero = true;
				} else if(allZero) {
					STHROW("VEDATensors does not support mixed-zero strides but found: " << strides);
				}
			}
		}

		return allZero;
	}()) {
		return {0, (size_t*)0, dtype(self), ptr(self)};
	}

	ASSERT(self.is_contiguous());
	return {sizes.size(), sizes, dtype(self), ptr(self)};
}

//------------------------------------------------------------------------------
#include "__ns.h"

//------------------------------------------------------------------------------
namespace c10 {
	REGISTER_ALLOCATOR(DEVICE_TYPE, veda::pytorch::allocator());
}

//------------------------------------------------------------------------------