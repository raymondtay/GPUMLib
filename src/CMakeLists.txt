INCLUDE(FindCUDA)

if (BUILD_RAN_AND_RBF_NETWORKS)
	INCLUDE_DIRECTORIES($ENV{CULA_INC_PATH})
	
	if (CULA_R17_OR_ABOVE)
		set (CULA_LIBS cula_lapack_basic)
	else (CULA_R17_OR_ABOVE)
		set (CULA_LIBS cula_core cula_lapack)
	endif (CULA_R17_OR_ABOVE)

	if (CMAKE_SIZEOF_VOID_P EQUAL 8)
		link_directories($ENV{CULA_LIB_PATH_64})
	else (CMAKE_SIZEOF_VOID_P EQUAL 8)
		link_directories($ENV{CULA_LIB_PATH_32})
	endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

	# RBF library
	CUDA_ADD_LIBRARY(GPUMLibRBF STATIC 
		RBF/RadialBasisFunction.cpp
		RBF/utils.cpp
		RBF/RBFKernels.cu
		KMeans/KMeans.cpp
		KMeans/KMeansKernels.cu
	)

	# RAN library
	CUDA_ADD_LIBRARY(GPUMLibRAN STATIC 
		RAN/ResourceAllocatingNetwork.cpp 
		RAN/RANKernels.cu
		RAN/utils.cpp
	)
endif (BUILD_RAN_AND_RBF_NETWORKS)

# BP and MBP library
CUDA_ADD_LIBRARY(GPUMLibMBP STATIC
	MBP/BackPropagation.cu 
	MBP/CalculateRMS.cu 
	MBP/CorrectWeightsKernel.cu 
	MBP/FireLayerKernel.cu 
	MBP/FireLayerNeuronsKernel.cu 
	MBP/LocalGradientKernel.cu 
	MBP/MultipleBackPropagation.cu 
	MBP/RobustLearning.cu 
	MBP/SelectiveInputs.cu 
	MBP/CalcLocalGradSelectiveInputs.cu 
	MBP/CorrectWeightsSelInputs.cu
)

# NMF library
CUDA_ADD_LIBRARY(GPUMLibNMF STATIC 
	NMF/NMFkernels.cu 
	NMF/NMFmultiplicativeDivergence.cu 
	NMF/NMFmultiplicativeEuclidian.cu 	
	NMF/NMFadditiveDivergence.cu
	NMF/NMFadditiveEuclidian.cu
	NMF/NMFadditiveDivergenceKernels.cu 
	NMF/NMFquality.cu
)

# Utils (Reduction, Random)
set (CU_FILES_Reduction 
	reduction/reduction.cpp 
	reduction/MaxKernel.cu 
	reduction/MinKernel.cu	
	reduction/SumKernel.cu
)

CUDA_ADD_LIBRARY(GPUMLibUtils STATIC 
	${CU_FILES_Reduction}
	random/random.cu
)

# RBM and DBN
CUDA_ADD_LIBRARY(GPUMLibDBN STATIC 
	RBM/rbm.cu 
	RBM/RBMinitKernels.cu 
	RBM/RBMstatus.cu 
	RBM/RBMstatusSmall.cu 
	RBM/RBMCorrectWeights.cu
)

target_link_libraries(GPUMLibDBN GPUMLibUtils)

# Examples and apps
add_subdirectory (examples)