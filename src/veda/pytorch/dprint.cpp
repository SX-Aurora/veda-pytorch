#include "api.h"

#include "__ns.h"
//------------------------------------------------------------------------------
template<typename T>
static void dprint_lst(std::ostringstream& ss, const T& lst) {
	ss << "[";
	bool isFirst = true;
	for(auto& value : lst) {
		if(isFirst)	isFirst = false;
		else		ss << ", ";
		dprint__(ss, value);
	}
	ss << "]";
}

//------------------------------------------------------------------------------
void dprint__(std::ostringstream& ss, const at::Scalar& value)	{	ss << value;						}
void dprint__(std::ostringstream& ss, const bool value)			{	ss << (value ? "true" : "false");	}
void dprint__(std::ostringstream& ss, const char* str)			{	ss << str;							}

//------------------------------------------------------------------------------
void dprint__(std::ostringstream& ss, const at::ITensorList& lst)							{	dprint_lst(ss, lst);				}
void dprint__(std::ostringstream& ss, const at::IntArrayRef& lst)							{	dprint_lst(ss, lst);				}
void dprint__(std::ostringstream& ss, const at::TensorList& lst)							{	dprint_lst(ss, lst);				}
void dprint__(std::ostringstream& ss, const std::vector<at::Tensor>& lst)					{	dprint_lst(ss, lst);				}
void dprint__(std::ostringstream& ss, const std::vector<int64_t>& lst)						{	dprint_lst(ss, lst);				}
void dprint__(std::ostringstream& ss, const std::vector<std::vector<at::Tensor>>& lst)		{	dprint_lst(ss, lst);				}
void dprint__(std::ostringstream& ss, const torch::List<std::optional<at::Tensor>>& lst)	{	dprint_lst(ss, lst.vec());			}

//------------------------------------------------------------------------------
void dprint__(std::ostringstream& ss, const std::optional<at::Tensor>& tensor) {
	if(tensor)	dprint__(ss, tensor.value());
	else		ss << "[#N/A]";
}

//------------------------------------------------------------------------------
void dprint__(std::ostringstream& ss, const at::Tensor& tensor) {
	ss << "Tensor[dtype=";
	switch(tensor.scalar_type()) {
		case c10::ScalarType::Byte:				ss << "U8";			break;
		case c10::ScalarType::Char:				ss << "S8";			break;
		case c10::ScalarType::Short:			ss << "S16";		break;
		case c10::ScalarType::Int:				ss << "S32";		break;
		case c10::ScalarType::Long:				ss << "S64";		break;
		case c10::ScalarType::Float:			ss << "F32";		break;
		case c10::ScalarType::Double:			ss << "F64";		break;
		case c10::ScalarType::Bool:				ss << "S8";			break;
		case c10::ScalarType::ComplexFloat:		ss << "F32_F32";	break;
		case c10::ScalarType::ComplexDouble:	ss << "F64_F64";	break;
	}
	ss << ", shape=";
	dprint_lst(ss, tensor.sizes());
	ss << ", strides=";
	dprint_lst(ss, tensor.strides());
	ss << ", device=" << tensor.device() << ", ptr=" << tensor.data_ptr() << "]";
}

//------------------------------------------------------------------------------
#include "__ns.h"