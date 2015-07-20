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

//! \addtogroup rbfkernels Radial Basis Functions Network kernels
//! @{

#ifndef RBFkernels_h
#define RBFkernels_h

#include "../common/CudaDefinitions.h"
#include "../memory/DeviceArray.h"
#include "../memory/DeviceMatrix.h"

/*Define functions to call CUDA kernels in C program*/

//! Kernel that estimates the widths for each neuron in the hidden layer.
//! \param[in] Distance Matrix with the distance between all centers.
//! \param[in] distance_height Height of the distance matrix.
//! \param[in] distance_width Width of the distance matrix.
//! \param[in] rneighbours Number of neighbours to use in width estimation.
//! \param[out] widths Array with the widths calculated for each neuron.
extern "C" void KernelAdjustWidths(cudafloat *Distance, int distance_height, int distance_width, int rneighbours, float *widths);

//! Kernel that calculates the distance between all rows of matrix A in relation to all rows of matrix B, the result is stored in matrix C. Index (i,j) in matrix C is equivalent to the distance between row i in matrix A and row j in matrix B.
//! \param[out] d_C Final matrix with the calculated distances.
//! \param[in] d_A Matrix A.
//! \param[in] d_B Matrix B.
//! \param[in] uiWA Width of matrix A.
//! \param[in] uiWB Width of matrix B.
//! \param[in] uiWC Width of matrix C.
//! \param[in] uiHC Height of matrix C.
extern "C" void KernelCalculateDistance(cudafloat *d_C, cudafloat* d_A, cudafloat* d_B,int uiWA,int uiWB, int uiWC, int uiHC);

//! Kernel to calculates the activation of the hidden layer, every row in matrix A (inputs) in relation to all rows in matrix B (hidden layer neurons). 
//! \param[out] d_C Final matrix with the activation values.
//! \param[in] d_A Matrix A with the training inputs.
//! \param[in] d_B Matrix B with the center values of the hidden layer neurons.
//! \param[in] uiWA Width of matrix A.
//! \param[in] uiWB Width of matrix B.
//! \param[in] uiWC Width of matrix C.
//! \param[in] uiHC Height of matrix C.
//! \param[in] scalingfactor Scaling factor applied to the widths.
//! \param[in] c_width Widths of the gaussian function applied by the hidden layer neurons.
extern "C" void KernelActivationMatrix(cudafloat *d_C, cudafloat* d_A, cudafloat* d_B,int uiWA,int uiWB, int uiWC, int uiHC, float scalingfactor, float* c_width);

//! Kernel that calculates the inverse of the values in array S and stores the result as diagonal matrix in Output.
//! \param[out] Output Diagonal matrix with the inverse of the values in S.
//! \param[in] output_height Height of the output matrix.
//! \param[in] output_width Width of the output matrix.
//! \param[in] S Array with the singular values of the matrix decomposition, obtained via SVD.
extern "C" void KernelSigmaInverse(float *Output, int output_width, int output_height, cudafloat *S);
#endif

//! @}