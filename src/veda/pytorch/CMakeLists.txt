LIST(APPEND SRC
	${CMAKE_CURRENT_LIST_DIR}/Allocator.cpp
	${CMAKE_CURRENT_LIST_DIR}/Guard.cpp
	${CMAKE_CURRENT_LIST_DIR}/op_binary.cpp
	${CMAKE_CURRENT_LIST_DIR}/op_bitwise.cpp
	${CMAKE_CURRENT_LIST_DIR}/op_cat.cpp
	${CMAKE_CURRENT_LIST_DIR}/op_complex.cpp
	${CMAKE_CURRENT_LIST_DIR}/op_copy.cpp
	${CMAKE_CURRENT_LIST_DIR}/op_cumsum.cpp
	${CMAKE_CURRENT_LIST_DIR}/op_fill.cpp
	${CMAKE_CURRENT_LIST_DIR}/op_masked_fill.cpp
	${CMAKE_CURRENT_LIST_DIR}/op_masked_select.cpp
	${CMAKE_CURRENT_LIST_DIR}/op_mean.cpp
	${CMAKE_CURRENT_LIST_DIR}/op_norm.cpp
	${CMAKE_CURRENT_LIST_DIR}/op_reduce.cpp
	${CMAKE_CURRENT_LIST_DIR}/op_tensor.cpp
	${CMAKE_CURRENT_LIST_DIR}/op_unary.cpp
)
