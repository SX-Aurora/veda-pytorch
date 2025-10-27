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
c10::DeviceIndex	VEGuardImpl::deviceCount		(void) const noexcept			{	return m_deviceCnt;										}
c10::DeviceType		VEGuardImpl::type				(void) const					{	return DEVICE_TYPE;										}
c10::Stream			VEGuardImpl::exchangeStream		(c10::Stream s) const noexcept	{	return c10::Stream(c10::Stream::DEFAULT, getDevice());	}
c10::Stream			VEGuardImpl::getStream			(c10::Device d) const noexcept	{	return c10::Stream(c10::Stream::DEFAULT, getDevice());	}
void				VEGuardImpl::setDefaultIdx		(const int idx)					{	m_defaultIdx = idx;										}
void				VEGuardImpl::uncheckedSetDevice	(c10::Device d) const noexcept	{	setDevice(d);											}

//------------------------------------------------------------------------------
VEGuardImpl::VEGuardImpl(void) :
	m_defaultIdx(0),
	m_exitVEDA([] {
		auto res = vedaInit(0);
		if		(res == VEDA_ERROR_ALREADY_INITIALIZED)	return false;
		else if	(res == VEDA_SUCCESS)					return true;
		else											CVEDA(res);
		return false;
	}()),
	m_deviceCnt([] {
		int cnt = 0;
		CVEDA(vedaDeviceGetCount(&cnt));
		return cnt;
	}())
{
	assert(s_guard == 0);
	s_guard = this;
}

//------------------------------------------------------------------------------
VEGuardImpl::~VEGuardImpl(void) {
	for(auto [idx, ctx] : m_ctxs)
		CVEDA(vedaDevicePrimaryCtxRelease(idx));
	if(m_exitVEDA)
		CVEDA(vedaExit());
}

//------------------------------------------------------------------------------
c10::Device VEGuardImpl::exchangeDevice(c10::Device d) const {
	auto o = getDevice();
	setDevice(d);
	return o;
}

//------------------------------------------------------------------------------
void VEGuardImpl::syncAll(void) const {
	for(auto [idx, ctx] : m_ctxs) {
		CVEDA(vedaCtxPushCurrent(ctx));	
		VEDAcontext ctx_;
		CVEDA(vedaCtxPopCurrent(&ctx_));
	}
}

//------------------------------------------------------------------------------
VEDAcontext VEGuardImpl::getCTX(int idx) {
	if(idx == -1)
		idx = m_defaultIdx;
		
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
	VEDAdevice idx = -1;
	auto res = vedaCtxGetDevice(&idx);
	if(res == VEDA_ERROR_UNKNOWN_CONTEXT)	idx = m_defaultIdx;
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
	CVEDA(vedaCtxPopCurrent(&ctx));
}

//------------------------------------------------------------------------------
void VEGuardImpl::setDevice(c10::Device d) const {
	// Requires non-const version of VEGuardImpl
	CVEDA(vedaCtxSetCurrent(getGuardImpl()->getCTX(d.index())));
}

//------------------------------------------------------------------------------
void sync(const int64_t idx) {
	GUARD(idx);
	CVEDA(vedaCtxSynchronize());
}

//------------------------------------------------------------------------------
void syncAll(void) {
	getGuardImpl()->syncAll();
}

//------------------------------------------------------------------------------
int64_t memoryAllocated(const int64_t idx) {
	GUARD(idx);
	size_t free = 0, total = 0;
	CVEDA(vedaMemGetInfo(&free, &total));
	return total - free;
}

//------------------------------------------------------------------------------
int64_t getCurrentDevice(void) {
	return getGuardImpl()->getDevice().index();
}

//------------------------------------------------------------------------------
int64_t deviceCount(void) {
	return getGuardImpl()->deviceCount();
}

//------------------------------------------------------------------------------
void setDevice(const int64_t idx) {
	getGuardImpl()->setDefaultIdx((int)idx);
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