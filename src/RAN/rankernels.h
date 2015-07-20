/*
	Ricardo Quintas is an MSc Student at the University of Coimbra, Portugal
    Copyright (C) 2009, 2010 Ricardo Quintas

	This file is part of GPUMLib.

    GPUMLib is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

//! \addtogroup rankernels Resource Allocating Network kernels
//! @{

#ifndef RANkernels_h
#define RANkernels_h

#include "../common/CudaDefinitions.h"
#include "../memory/DeviceArray.h"
#include "../memory/DeviceMatrix.h"
#include "../memory/HostArray.h"
#include "../memory/HostMatrix.h"

/*Define functions to call CUDA kernels in C program*/
extern "C" void FindMin(cudafloat *Output, int output_height, int output_width, float *min_array, int* min_idx,cudafloat* Targets);
extern "C" void KernelEuclidianDistance(cudafloat *Output, int output_height, int output_width, cudafloat *Input, int input_width, cudafloat *Centers, int centers_width);

extern "C" void KernelFindNearestCenter(cudafloat *Output, int output_width, cudafloat *Sample, cudafloat *Centers, int centers_width, float* min_value);

extern "C" void KernelActivationMatrix(cudafloat *Output, int output_height, int output_width, cudafloat *Input, int input_width, cudafloat *Centers, int centers_width, float *c_width, float scalingfactor);

extern "C" void KernelSigmaInverse(float *Output, int output_width, int output_height, cudafloat *S);

extern "C" unsigned int nextPow2( unsigned int x );
extern "C" cudafloat KernelCalculateDistance(cudafloat *output, cudafloat *A, cudafloat *B,int n);

extern "C" void matmul(cudafloat *d_C, cudafloat* d_A, cudafloat* d_B,int uiWA,int uiWB, int uiWC, int uiHC);

extern "C" void KernelCalculateNetworkActivation(cudafloat* output, cudafloat* Sample,int Length,cudafloat* dCenters,int NumCenters,cudafloat* dWeights,int NumClasses,cudafloat* dWidths,float scaling_factor);

extern "C" void KernelUpdateWidths(cudafloat* dWidths, cudafloat* newWidths, int Length);


extern "C" void KernelCalculateError(cudafloat* result, cudafloat* target, cudafloat* output, int Length, float* error);

extern "C" void KernelSumActivations(cudafloat* output, int Length, int NumCenters);





extern "C" void KernelCopyTo(cudafloat* dCenters, cudafloat *Sample,int Length);

#endif


//! @}