#include "__ns.h"
//------------------------------------------------------------------------------
class VEGuardImpl final : public c10::impl::DeviceGuardImplInterface {
	
			std::mutex					m_mutex;
			std::map<int, VEDAcontext>	m_ctxs;
			int							m_defaultIdx;
	const	bool						m_exitVEDA;
	const	int							m_deviceCnt;

			VEDAcontext			getCTX				(int idx);
public:
								VEGuardImpl			(void);
								~VEGuardImpl		(void);
			void				pop					(void) const;
			void				push				(const int idx);
			void				syncAll				(void) const;
			void				setDefaultIdx		(const int idx);
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
VEGuardImpl* getGuardImpl(void);

//------------------------------------------------------------------------------
#define GUARD(T) const VEGuard __guard__(T)

//------------------------------------------------------------------------------
class VEGuard final {
	const c10::Device m_device;
	const c10::Device m_prevDevice;

	inline void init(void) const {
		ASSERT(m_device.type() == DEVICE_TYPE);
		getGuardImpl()->push(m_device.index());
	}

public:
	inline VEGuard(const c10::Device device)			: m_device(device),					m_prevDevice(getGuardImpl()->exchangeDevice(m_device))	{ init(); }
	inline VEGuard(const at::Tensor& self)				: m_device(self.device()),			m_prevDevice(getGuardImpl()->exchangeDevice(m_device))	{ init(); }
	inline VEGuard(const at::TensorList& list)			: m_device(list.front().device()),	m_prevDevice(getGuardImpl()->exchangeDevice(m_device))	{ init(); }
	inline VEGuard(const at::ITensorListRef& list)		: m_device(list.front().device()),	m_prevDevice(getGuardImpl()->exchangeDevice(m_device))	{ init(); }
	inline VEGuard(const at::TensorOptions& options)	: m_device(options.device()),		m_prevDevice(getGuardImpl()->exchangeDevice(m_device))	{ init(); }
	inline VEGuard(const c10::DeviceIndex device)		: m_device({DEVICE_TYPE, device}),	m_prevDevice(getGuardImpl()->exchangeDevice(m_device))	{ init(); }
	inline VEGuard(const c10::TensorImpl* self)			: m_device(self->device()),			m_prevDevice(getGuardImpl()->exchangeDevice(m_device))	{ init(); }
	
	inline ~VEGuard(void) {
		getGuardImpl()->pop();
	}
};

//------------------------------------------------------------------------------
int64_t	deviceCount		(void);
int64_t	getCurrentDevice(void);
int64_t	memoryAllocated	(const int64_t idx);
void	setDevice		(const int64_t idx);
void	sync			(const int64_t idx);
void	syncAll			(void);

//------------------------------------------------------------------------------
#include "__ns.h"