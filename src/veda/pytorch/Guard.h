#include "__ns.h"
//------------------------------------------------------------------------------
#define GUARD(T) const VEGuard __guard__(T)

//------------------------------------------------------------------------------
class VEGuard final {
	const c10::Device	m_device;

	inline void init(void) const {
		ASSERT(m_device.type() == DEVICE_TYPE);
		auto idx = m_device.index();
		VEDAcontext ctx;
		if(idx >= 0) {
			CVEDA(vedaDevicePrimaryCtxRetain(&ctx, m_device.index()));
		} else if(idx == -1) {
			if(vedaCtxGetCurrent(&ctx) != VEDA_SUCCESS)
				CVEDA(vedaDevicePrimaryCtxRetain(&ctx, 0));
		} else {
			THROW("Illegal device index: %i", idx);
		}
		CVEDA(vedaCtxPushCurrent(ctx));
	}

public:
	inline VEGuard(const c10::Device device)			: m_device(device)					{	init();	}
	inline VEGuard(const at::Tensor& self)				: m_device(self.device())			{	init();	}
	inline VEGuard(const at::TensorList& list)			: m_device(list.front().device())	{	init();	}
	inline VEGuard(const at::TensorOptions& options)	: m_device(options.device())		{	init();	}
	inline VEGuard(const c10::DeviceIndex device)		: m_device({DEVICE_TYPE, device})	{	init();	}
	inline VEGuard(const c10::TensorImpl* self)			: m_device(self->device())			{	init();	}
	
	inline ~VEGuard(void) {
		VEDAcontext ctx;
		CVEDA(vedaCtxPopCurrent(&ctx));
	}
};

//------------------------------------------------------------------------------
struct VEGuardImpl final : public c10::impl::DeviceGuardImplInterface {
								VEGuardImpl			(void);
	virtual	c10::Device			exchangeDevice		(c10::Device d) const override;
	virtual	c10::Device			getDevice			(void) const override;
	virtual	c10::DeviceIndex	deviceCount			(void) const noexcept override;
	virtual	c10::DeviceType		type				(void) const override;
	virtual	c10::Stream			exchangeStream		(c10::Stream s) const noexcept override;
	virtual	c10::Stream			getStream			(c10::Device d) const noexcept override;
	virtual	void				setDevice			(c10::Device d) const override;
	virtual	void				uncheckedSetDevice	(c10::Device d) const noexcept override;
};

//------------------------------------------------------------------------------
const VEGuardImpl* getGuardImpl(void);

//------------------------------------------------------------------------------
#include "__ns.h"