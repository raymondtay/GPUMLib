include(FindCUDA)
add_definitions (-D_CONSOLE)

CUDA_ADD_EXECUTABLE(ATS ATS.cpp)
target_link_libraries(ATS GPUMLibMBP)
install(TARGETS ATS DESTINATION ${CMAKE_BINARY_DIR})