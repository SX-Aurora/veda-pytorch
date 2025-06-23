#include "api.h"

#include "__ns.h"
//------------------------------------------------------------------------------
// Static
//------------------------------------------------------------------------------
static VEGuardImpl* s_guard = 0;

//------------------------------------------------------------------------------
VEGuardImpl* getGuardImpl(void) {
	assert(s_guard);
	return s_guard;
}

//------------------------------------------------------------------------------
// VEGuardImpl
//------------------------------------------------------------------------------
c10::Device			VEGuardImpl::exchangeDevice		(c10::Device d) const			{	auto o = getDevice(); setDevice(d); return o;			}
c10::DeviceIndex	VEGuardImpl::deviceCount		(void) const noexcept			{	return m_deviceCnt;										}
c10::DeviceType		VEGuardImpl::type				(void) const					{	return DEVICE_TYPE;										}
c10::Stream			VEGuardImpl::exchangeStream		(c10::Stream s) const noexcept	{	return c10::Stream(c10::Stream::DEFAULT, getDevice());	}
c10::Stream			VEGuardImpl::getStream			(c10::Device d) const noexcept	{	return c10::Stream(c10::Stream::DEFAULT, getDevice());	}
void				VEGuardImpl::uncheckedSetDevice	(c10::Device d) const noexcept	{	setDevice(d);											}

//------------------------------------------------------------------------------
VEGuardImpl::VEGuardImpl(void) :
	m_deviceCnt(0),
	m_exitVEDA (false)
{
	assert(s_guard == 0);
	s_guard = this;
	auto res = vedaInit(0);
	if		(res == VEDA_ERROR_ALREADY_INITIALIZED)	m_exitVEDA = false;
	else if	(res == VEDA_SUCCESS)					m_exitVEDA = true;
	else											CVEDA(res);
	CVEDA(vedaDeviceGetCount(&m_deviceCnt));
}

//------------------------------------------------------------------------------
VEGuardImpl::~VEGuardImpl(void) {
	for(auto [idx, ctx] : m_ctxs)
		CVEDA(vedaDevicePrimaryCtxRelease(idx));
	if(m_exitVEDA)
		CVEDA(vedaExit());
}

//------------------------------------------------------------------------------
VEDAcontext VEGuardImpl::getCTX(int idx) {
	if(idx == -1)
		idx = 0;
		
	if(idx < 0 || idx >= m_deviceCnt)
		THROW("Device index needs to be between 0 and %i but is %i!", m_deviceCnt, idx);
	
	std::lock_guard<std::mutex> __lock(m_mutex);
	auto it = m_ctxs.find(idx);
	if(it != m_ctxs.end())
		return it->second;
	
	VEDAcontext ctx;
	CVEDA(vedaDevicePrimaryCtxRetain(&ctx, idx));
	m_ctxs.emplace(idx, ctx);
	return ctx;
}

//------------------------------------------------------------------------------
c10::Device VEGuardImpl::getDevice(void) const {
	VEDAdevice idx;
	auto res = vedaCtxGetDevice(&idx);
	if(res == VEDA_ERROR_UNKNOWN_CONTEXT)	idx = 0;
	else									CVEDA(res);
	return {DEVICE_TYPE, (c10::DeviceIndex)idx};
}

//------------------------------------------------------------------------------
void VEGuardImpl::push(const int idx) {
	CVEDA(vedaCtxPushCurrent(getCTX(idx)));
}

//------------------------------------------------------------------------------
void VEGuardImpl::pop(void) const {
	VEDAcontext ctx;
	CVEDA(vedaCtxSynchronize());
	CVEDA(vedaCtxPopCurrent(&ctx));
}

//------------------------------------------------------------------------------
void VEGuardImpl::setDevice(c10::Device d) const {
	// Requires non-const version of VEGuardImpl
	CVEDA(vedaCtxSetCurrent(getGuardImpl()->getCTX(d.index())));
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