#include "__ns.h"
//------------------------------------------------------------------------------
#define GUARD(T) const VEGuard __guard__(T)

//------------------------------------------------------------------------------
class VEGuard final {
	inline void init(const c10::Device d) {
		L_WARN("DEVICE: %i", d.index());
		L_WARN("TODO: Handle PyTorch default device shit!");
		VEDAcontext ctx;
		CVEDA(vedaDevicePrimaryCtxRetain(&ctx, d.index()));
		CVEDA(vedaCtxPushCurrent(ctx));
	}

public:
	inline VEGuard(const c10::Device device)			{	init(device);					}
	inline VEGuard(const at::Tensor& self)				{	init(self.device());			}
	inline VEGuard(const at::TensorList& list)			{	init(list.front().device());	}
	inline VEGuard(const at::TensorOptions& options)	{	init(options.device());			}
	inline VEGuard(const c10::DeviceIndex device)		{	init({DEVICE_TYPE, device});	}
	inline VEGuard(const c10::TensorImpl* self)			{	init(self->device());			}
	
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
#include "__ns.h"