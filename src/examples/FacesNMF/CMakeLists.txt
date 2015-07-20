include(FindCUDA)
ADD_DEFINITIONS(-D_AFXDLL)
set(CMAKE_MFC_FLAG 2)

CUDA_ADD_EXECUTABLE(FacesNMF WIN32 
	FacesNMF.cpp 
	FacesNMFDlg.cpp 
	FacesNMF.rc 
	stdafx.cpp
)

CUDA_ADD_CUBLAS_TO_TARGET(FacesNMF)
target_link_libraries(FacesNMF GPUMLibNMF)
install(TARGETS FacesNMF DESTINATION ${CMAKE_BINARY_DIR})
