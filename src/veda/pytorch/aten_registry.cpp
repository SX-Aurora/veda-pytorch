#include "api.h"

TORCH_LIBRARY(veda, m) {
	m.def("device_count",		veda::pytorch::deviceCount);
	m.def("get_current_device",	veda::pytorch::getCurrentDevice);
	m.def("memory_allocated",	veda::pytorch::memoryAllocated);
	m.def("set_device",			veda::pytorch::setDevice);
	m.def("sync",				veda::pytorch::sync);
	m.def("sync_all",			veda::pytorch::syncAll);
}