include(FindCUDA)
add_definitions (-D_CONSOLE)

CUDA_ADD_EXECUTABLE(MBP MBP.cpp MBPutils.cpp)
target_link_libraries(MBP GPUMLibMBP)
install(TARGETS MBP DESTINATION ${CMAKE_BINARY_DIR})