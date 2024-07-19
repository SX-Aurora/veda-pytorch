#include "api.h"

#include "__ns.h"
//------------------------------------------------------------------------------
static VEGuardImpl* s_guard = 0;
thread_local int t_currentDevice = 0;

//------------------------------------------------------------------------------
const VEGuardImpl* getGuardImpl(void) {
	assert(s_guard);
	return s_guard;
}

//------------------------------------------------------------------------------
c10::Device			VEGuardImpl::getDevice			(void) const					{	return {DEVICE_TYPE, (c10::DeviceIndex)(t_currentDevice)};	}
c10::DeviceIndex	VEGuardImpl::deviceCount		(void) const noexcept			{	return m_deviceCnt;											}
c10::DeviceType		VEGuardImpl::type				(void) const					{	return DEVICE_TYPE;											}
c10::Stream			VEGuardImpl::exchangeStream		(c10::Stream s) const noexcept	{	return c10::Stream(c10::Stream::DEFAULT, getDevice());		}
c10::Stream			VEGuardImpl::getStream			(c10::Device d) const noexcept	{	return c10::Stream(c10::Stream::DEFAULT, getDevice());		}
void				VEGuardImpl::setDevice			(c10::Device d) const			{	t_currentDevice = d.index();								}
void				VEGuardImpl::uncheckedSetDevice	(c10::Device d) const noexcept	{	setDevice(d);												}

//------------------------------------------------------------------------------
VEGuardImpl::VEGuardImpl(void) :
	m_deviceCnt(0)
{
	assert(s_guard == 0);
	s_guard = this;
	auto res = vedaInit(0);
	if(res != VEDA_SUCCESS && res != VEDA_ERROR_ALREADY_INITIALIZED)
		CVEDA(res);
	CVEDA(vedaDeviceGetCount(&m_deviceCnt));
}

//------------------------------------------------------------------------------
c10::Device VEGuardImpl::exchangeDevice(c10::Device d) const {
	auto old_device = getDevice();
	t_currentDevice = d.index();
	return old_device;
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