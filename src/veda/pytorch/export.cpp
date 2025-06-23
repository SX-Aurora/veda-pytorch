#include "api.h"

//------------------------------------------------------------------------------
extern "C" int veda_pytorch_get_current_device(void) {
	return veda::pytorch::getGuardImpl()->getDevice().index();
}

//------------------------------------------------------------------------------
extern "C" int veda_pytorch_device_count(void) {
	return veda::pytorch::getGuardImpl()->deviceCount();
}

//------------------------------------------------------------------------------
extern "C" void veda_pytorch_set_device(const int idx) {
	veda::pytorch::getGuardImpl()->setDevice({DEVICE_TYPE, (c10::DeviceIndex)(idx)});
}

//------------------------------------------------------------------------------
extern "C" int64_t veda_pytorch_memory_allocated(const int idx) {
	return veda::pytorch::memoryAllocated(idx);
}

//------------------------------------------------------------------------------
extern "C" void veda_pytorch_sync(const int idx) {
	return veda::pytorch::sync(idx);
}

//------------------------------------------------------------------------------