add_subdirectory (ATS)
add_subdirectory (BP)
add_subdirectory (MBP)

if (BUILD_RAN_AND_RBF_NETWORKS)
	add_subdirectory (RAN)
	add_subdirectory (RBF)
endif (BUILD_RAN_AND_RBF_NETWORKS)

add_subdirectory (DBN)
add_subdirectory (reduction)
add_subdirectory (SVM)

if (MSVC)
	add_subdirectory (FacesNMF)
ENDIF(MSVC)