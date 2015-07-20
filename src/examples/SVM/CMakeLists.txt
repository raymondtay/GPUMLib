include(FindCUDA)
#add_definitions (-D_CONSOLE)

CUDA_ADD_EXECUTABLE(svm svm_example.cu)
install(TARGETS svm DESTINATION ${CMAKE_BINARY_DIR})