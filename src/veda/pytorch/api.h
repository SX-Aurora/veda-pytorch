#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <c10/core/DispatchKey.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <ATen/core/op_registration/op_registration.h>	// RegisterOperators
#include <ATen/native/Resize.h>							// requires PyTorch Source!
#include <ATen/native/Copy.h>							// requires PyTorch Source!
#include <ATen/native/UnaryOps.h>						// requires PyTorch Source!
#include <ATen/native/Fill.h>							// requires PyTorch Source!
#include <ATen/native/BinaryOps.h>						// requires PyTorch Source!
#include <ATen/native/TensorIterator.h>					// requires PyTorch Source!
#include <ATen/native/PointwiseOps.h>					// requires PyTorch Source!
#include <ATen/native/TensorAdvancedIndexing.h>			// requires PyTorch Source!
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>
#include <torch/csrc/api/include/torch/version.h>
#pragma GCC diagnostic pop

#define DEVICE_TYPE		c10::DeviceType::VE
#define DEVICE_TYPE_	VE
#define DISPATCH_KEY	c10::DispatchKey::VE

#define L_MODULE "VEDA/PyTorch"
#include <tungl/c.h>
#include <veda/api.h>
#include <veda/cpp/api.h>
#include <veda/tensors/api.h>
#undef CVEDA
#define CVEDA(...) try {\
	veda::pytorch::check(__VA_ARGS__, __FILE__, __LINE__);\
} catch(const veda::cpp::Exception& e) {\
	THROWAT(L_MODULE, e.file().data(), e.line(), "VEDA_ERROR: %s", e.what().data());\
}

#define STHROWAT(FILE, LINE, ...) {				\
	std::ostringstream g;						\
	g << __VA_ARGS__;							\
	const auto msg = g.str();					\
	THROWAT(L_MODULE, FILE, LINE, msg.c_str());	\
}
#define STHROW(...) STHROWAT(__FILE__, __LINE__, __VA_ARGS__)

#include "__ns.h"
//------------------------------------------------------------------------------
inline void check(VEDAresult res, const char* file, const int line) {
	if(__builtin_expect((res != VEDA_SUCCESS), 0)) {
		const char* err;
		vedaGetErrorName(res, &err);
		THROWAT(L_MODULE, file, line, "VEDA_ERROR: %s", err);
	}
}

//------------------------------------------------------------------------------
#include "__ns.h"

#define TORCH_VERSION_ ((TORCH_VERSION_MAJOR * 10000) + (TORCH_VERSION_MINOR * 100) + (TORCH_VERSION_PATCH))

#include "Guard.h"
#include "Allocator.h"
#include "dprint.h"
