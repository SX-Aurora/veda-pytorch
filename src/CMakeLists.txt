INCLUDE(${CMAKE_CURRENT_LIST_DIR}/veda/pytorch/CMakeLists.txt)
INCLUDE(${CMAKE_CURRENT_LIST_DIR}/veda/pytorch/c10d/CMakeLists.txt)
INCLUDE(${CMAKE_CURRENT_LIST_DIR}/python/CMakeLists.txt)
ADD_PYTHON_WHEEL(dist ${CMAKE_CURRENT_LIST_DIR}/veda-pytorch.json)