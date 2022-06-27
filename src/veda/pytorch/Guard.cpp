#include "api.h"

#include "__ns.h"
//------------------------------------------------------------------------------
static VEGuardImpl* s_guard = 0;

//------------------------------------------------------------------------------
const VEGuardImpl* getGuardImpl(void) {
	assert(s_guard);
	return s_guard;
}

//------------------------------------------------------------------------------
c10::DeviceType		VEGuardImpl::type				(void) const					{	return DEVICE_TYPE;										}
c10::Stream			VEGuardImpl::exchangeStream		(c10::Stream s) const noexcept	{	return c10::Stream(c10::Stream::DEFAULT, getDevice());	}
c10::Stream			VEGuardImpl::getStream			(c10::Device d) const noexcept	{	return c10::Stream(c10::Stream::DEFAULT, getDevice());	}
void				VEGuardImpl::uncheckedSetDevice	(c10::Device d) const noexcept	{	setDevice(d);											}

//------------------------------------------------------------------------------
VEGuardImpl::VEGuardImpl(void) {
	assert(s_guard == 0);
	s_guard = this;
	auto res = vedaInit(0);
	if(res != VEDA_SUCCESS && res != VEDA_ERROR_ALREADY_INITIALIZED)
		CVEDA(res);
}

//------------------------------------------------------------------------------
c10::DeviceIndex VEGuardImpl::deviceCount(void) const noexcept {
	int cnt = 0;
	CVEDA(vedaDeviceGetCount(&cnt));
	return cnt;
}

//------------------------------------------------------------------------------
c10::Device VEGuardImpl::exchangeDevice(c10::Device d) const {
	auto old_device = getDevice();
	if(old_device.index() != d.index())
		setDevice(d);
	return old_device;
}

//------------------------------------------------------------------------------
c10::Device VEGuardImpl::getDevice(void) const {
	VEDAdevice device;
	auto res = vedaCtxGetDevice(&device);
	if(res == VEDA_ERROR_UNKNOWN_CONTEXT)	device = 0;
	else									CVEDA(res);
	return {DEVICE_TYPE, (c10::DeviceIndex)(device)};
}

//------------------------------------------------------------------------------
void VEGuardImpl::setDevice(c10::Device d) const {
	VEDAcontext ctx;
	CVEDA(vedaDevicePrimaryCtxRetain(&ctx, d.index()));
	CVEDA(vedaCtxSetCurrent(ctx));
}

//------------------------------------------------------------------------------
#include "__ns.h"

namespace at {
	namespace ve {
		namespace detail {
			C10_REGISTER_GUARD_IMPL(DEVICE_TYPE_, veda::pytorch::VEGuardImpl);
		}
	}
}