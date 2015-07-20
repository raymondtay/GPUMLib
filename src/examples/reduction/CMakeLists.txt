include(FindCUDA)
add_definitions (-D_CONSOLE)

CUDA_ADD_EXECUTABLE(reductionapp reductionapp.cpp)
target_link_libraries(reductionapp GPUMLibUtils)
install(TARGETS reductionapp DESTINATION ${CMAKE_BINARY_DIR})
