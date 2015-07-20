include(FindCUDA)
add_definitions (-D_CONSOLE)	

CUDA_ADD_EXECUTABLE(BP BP.cpp ../MBP/MBPutils.cpp)
target_link_libraries(BP GPUMLibMBP)
install(TARGETS BP DESTINATION ${CMAKE_BINARY_DIR})