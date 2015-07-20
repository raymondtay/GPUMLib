include(FindCUDA)
add_definitions (-D_CONSOLE)

CUDA_ADD_EXECUTABLE(DBN DBNapp.cpp)
target_link_libraries(DBN GPUMLibDBN ${CUDA_curand_LIBRARY})
install(TARGETS DBN DESTINATION ${CMAKE_BINARY_DIR})
