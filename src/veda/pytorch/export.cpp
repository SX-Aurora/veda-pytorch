#include "api.h"

//------------------------------------------------------------------------------
extern "C" int veda_pytorch_get_current_device(void) {
	return veda::pytorch::getGuardImpl()->getDevice().index();
}

//------------------------------------------------------------------------------