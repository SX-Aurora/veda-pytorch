#include "__ns.h"
//------------------------------------------------------------------------------
void dprint__(std::ostringstream& ss, const at::ITensorList& lst);
void dprint__(std::ostringstream& ss, const at::IntArrayRef& lst);
void dprint__(std::ostringstream& ss, const at::Scalar& value);
void dprint__(std::ostringstream& ss, const at::Tensor& tensor);
void dprint__(std::ostringstream& ss, const at::TensorList& lst);
void dprint__(std::ostringstream& ss, const bool value);
void dprint__(std::ostringstream& ss, const char* str);
void dprint__(std::ostringstream& ss, const std::vector<int64_t>& lst);
void dprint__(std::ostringstream& ss, const std::optional<at::Tensor>& tensor);
void dprint__(std::ostringstream& ss, const std::vector<at::Tensor>& lst);
void dprint__(std::ostringstream& ss, const std::vector<std::vector<at::Tensor>>& lst);
void dprint__(std::ostringstream& ss, const torch::List<std::optional<at::Tensor>>& lst);

//------------------------------------------------------------------------------
template<typename T>
inline typename std::enable_if<std::is_fundamental<T>::value>::type dprint__(std::ostringstream& ss, const T& value) {
	ss << value;
}

//------------------------------------------------------------------------------
template<typename T, typename... Args>
inline void dprint_(std::ostringstream& ss, const T& t, const Args&... args) {
	dprint__(ss, t);
	if constexpr(sizeof...(Args)) {
		ss << ", ";
		dprint_(ss, args...);
	}
}

//------------------------------------------------------------------------------
template<typename T, typename... Args>
inline void dprint(const T& t, const Args&... args) {
	constexpr auto level = TUNGL_LEVEL_TRACE;
	if(tungl_is_active(level)) {
		std::ostringstream ss;
		dprint__(ss, t);
		ss << "(";
		if constexpr(sizeof...(Args))
			dprint_(ss, args...);
		ss << ")";
		auto str = ss.str();
		tungl_log(level, L_MODULE, __FILE__, __LINE__, str.c_str());
	}
}

//------------------------------------------------------------------------------
#include "__ns.h"