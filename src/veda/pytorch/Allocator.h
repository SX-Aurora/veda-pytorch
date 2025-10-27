#include "__ns.h"
//------------------------------------------------------------------------------
VEDATensors_dtype		dtype					(const at::Tensor& self);
VEDATensors_dtype		dtype					(const c10::TensorImpl* self);
VEDATensors_handle		handle					(const at::Tensor& self);
VEDATensors_handle		handle					(void);
VEDATensors_reduce_op	reduction				(int64_t reduction);
VEDATensors_scalar		scalar					(const c10::ScalarType& type, const c10::Scalar& value);
VEDATensors_tensor		py2veda					(const at::Tensor& self);
at::Allocator*			allocator				(void);
at::Scalar				toPyScalar				(const c10::ScalarType& type, const VEDATensors_scalar value);
at::Tensor				empty					(at::IntArrayRef size, c10::optional<c10::ScalarType> dtype, c10::optional<c10::Layout> layout, c10::optional<c10::Device> device, c10::optional<bool> pin_memory, const c10::optional<c10::MemoryFormat> memory_format);
at::Tensor				sameDevice				(const at::Tensor& self, at::Tensor other);
at::Tensor				sameType				(const at::Tensor& self, at::Tensor other);
at::Tensor				toType					(at::Tensor tensor, const c10::ScalarType dtype);
bool					isBool					(const at::Tensor& self);
bool					isBool					(const c10::TensorImpl* self);
c10::TensorImpl*		resizePyTensor			(c10::TensorImpl* self, at::IntArrayRef size, c10::optional<at::IntArrayRef> stride);
size_t					numel					(const at::Tensor& self);

//------------------------------------------------------------------------------
inline at::Tensor empty_as(at::IntArrayRef sizes, const at::Tensor& self, const c10::ScalarType& dtype) {
	return empty(sizes, dtype, self.layout(), self.device(), false, at::MemoryFormat::Contiguous);
}

//------------------------------------------------------------------------------
inline	VEDAdeviceptr		ptr		(c10::TensorImpl* self)									{	return (VEDAdeviceptr)self->data();					}
inline	VEDAdeviceptr		ptr		(const at::Tensor& self)								{	return ptr(self.unsafeGetTensorImpl());				}
inline	at::Tensor			empty_as(at::IntArrayRef sizes, const at::Tensor& self)			{	return empty_as(sizes, self, self.scalar_type());	}
inline	at::Tensor			empty_as(const at::Tensor& self)								{	return empty_as(self.sizes(), self);				}
inline	at::Tensor			empty_as(const at::Tensor& self, const c10::ScalarType& dtype)	{	return empty_as(self.sizes(), self, dtype);			}
inline	c10::TensorImpl*	deref	(void* ref)												{	return (c10::TensorImpl*)ref;						}

//------------------------------------------------------------------------------
inline at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
	return empty(size, dtype, layout, device, pin_memory, c10::nullopt);
}

//------------------------------------------------------------------------------
inline at::Tensor toScalarPyTensor(const VEDATensors_scalar& value, const at::Tensor& self) {
	return at::scalar_tensor(toPyScalar(self.scalar_type(), value), at::device(self.device()).dtype(self.scalar_type()));
}

//------------------------------------------------------------------------------
inline at::Tensor wrapped_scalar_tensor(const at::Tensor& self, at::Scalar scalar) {
	auto tensor = at::scalar_tensor(scalar, at::device(self.device()).dtype(self.scalar_type()));
	tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
	return tensor;
}

//------------------------------------------------------------------------------
#include "__ns.h"