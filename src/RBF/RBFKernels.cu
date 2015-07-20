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
#include "../common/CudaDefinitions.h"
#include "rbfkernels.h"

KERNEL AdjustWidths(cudafloat *Distance, int distance_height, int distance_width, int rneighbours, float *widths){

	int idny = blockIdx.y*blockDim.y + threadIdx.y;

	float min_tmp = Distance[idny*distance_width];
	int idx = 0;

	float width = 0;
	int count = 0;

	if(idny < distance_height){

		while(count <= rneighbours ){

			min_tmp = -1;

			for(int j = 0; j < distance_width; j++){
				if((Distance[idny*distance_width+j] != -1 && Distance[idny*distance_width+j] < min_tmp) || min_tmp == -1){
					min_tmp = Distance[idny*distance_width+j];
					idx = j;
				}
			}

			Distance[idny*distance_width+idx] = -1;

			if(min_tmp != -1){
				width = width + pow(min_tmp,2);
				count = count + 1;
			}

		}

		widths[idny] = width/rneighbours;
	}
}

extern "C" void KernelAdjustWidths(cudafloat *Distance, int distance_height, int distance_width, int rneighbours, float *widths)
{
	int blockSize = 16;

	int wBlocks = 1;
	int hBlocks = distance_height/blockSize + ((distance_height%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	AdjustWidths<<<grid,threads>>>(Distance, distance_height, distance_width, rneighbours, widths);
}





KERNEL SigmaInverse(float *Output,  int output_height,int output_width, cudafloat *S){

	int idny = blockIdx.y*blockDim.y + threadIdx.y;

	//column-major
	if(idny < output_height)
		if(S[idny] != 0)
			Output[idny * output_height + idny] = 1/S[idny];

}

extern "C" void KernelSigmaInverse(float *Output, int output_height, int output_width, cudafloat *S)
{
	int blockSize = 16;

	int wBlocks = 1;
	int hBlocks = output_height/blockSize + ((output_height%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);

	SigmaInverse<<<grid,threads>>>(Output, output_height, output_width, S);
}





/*Adapted from the matrix multiplication examples of the CUDA Toolkit*/
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]

#define BLOCK_SIZE 16

__global__ void
CalculateDistance( float* C, float* A, float* B, int wA, int wB, int wC, int hC)
{

	int idnx = blockIdx.x*blockDim.x + threadIdx.x;
	int idny = blockIdx.y*blockDim.y + threadIdx.y;

	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd   = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep  = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = wB * BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep  = BLOCK_SIZE;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep) {

			// Declaration of the shared memory array As used to
			// store the sub-matrix of A
			__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

			// Declaration of the shared memory array Bs used to
			// store the sub-matrix of B
			__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

			// Load the matrices from device memory
			// to shared memory; each thread loads
			// one element of each matrix
			AS(ty, tx) = A[a + wA * ty + tx];
			BS(ty, tx) = B[b + wB * ty + tx];

			// Synchronize to make sure the matrices are loaded
			__syncthreads();

			// Calculate the distance between the two matrices;
			// each thread computes one element
			// of the block sub-matrix
			// for (int k = 0; k < BLOCK_SIZE && (a+k) <= aEnd; ++k)
			//     Csub += pow(AS(ty, k) - BS(tx, k),2);

			if((a+0) <= aEnd) Csub += pow(AS(ty, 0) - BS(tx, 0),2);
			if((a+1) <= aEnd) Csub += pow(AS(ty, 1) - BS(tx, 1),2);
			if((a+2) <= aEnd) Csub += pow(AS(ty, 2) - BS(tx, 2),2);
			if((a+3) <= aEnd) Csub += pow(AS(ty, 3) - BS(tx, 3),2);
			if((a+4) <= aEnd) Csub += pow(AS(ty, 4) - BS(tx, 4),2);
			if((a+5) <= aEnd) Csub += pow(AS(ty, 5) - BS(tx, 5),2);
			if((a+6) <= aEnd) Csub += pow(AS(ty, 6) - BS(tx, 6),2);
			if((a+7) <= aEnd) Csub += pow(AS(ty, 7) - BS(tx, 7),2);
			if((a+8) <= aEnd) Csub += pow(AS(ty, 8) - BS(tx, 8),2);
			if((a+9) <= aEnd) Csub += pow(AS(ty, 9) - BS(tx, 9),2);
			if((a+10) <= aEnd) Csub += pow(AS(ty, 10) - BS(tx, 10),2);
			if((a+11) <= aEnd) Csub += pow(AS(ty, 11) - BS(tx, 11),2);
			if((a+12) <= aEnd) Csub += pow(AS(ty, 12) - BS(tx, 12),2);
			if((a+13) <= aEnd) Csub += pow(AS(ty, 13) - BS(tx, 13),2);
			if((a+14) <= aEnd) Csub += pow(AS(ty, 14) - BS(tx, 14),2);
			if((a+15) <= aEnd) Csub += pow(AS(ty, 15) - BS(tx, 15),2);

			// Synchronize to make sure that the preceding
			// computation is done before loading two new
			// sub-matrices of A and B in the next iteration
			__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	if(idnx < wC && idny < hC){
		int c = wC * BLOCK_SIZE * by + BLOCK_SIZE * bx;
		C[c + wC * ty + tx] = sqrt(Csub);
	}
}

extern "C" void KernelCalculateDistance(cudafloat *d_C, cudafloat* d_A, cudafloat* d_B,int uiWA,int uiWB, int uiWC, int uiHC)
{

	int blockSize = 16;

	int wBlocks = uiWC/blockSize + ((uiWC%blockSize == 0)?0:1);
	int hBlocks = uiHC/blockSize + ((uiHC%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);

	CalculateDistance<<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB, uiWC, uiHC);

}


__global__ void
ActivationMatrix( float* C, float* A, float* B, int wA, int wB, int wC, int hC,float scalingfactor, float* c_width)
{

	int idnx = blockIdx.x*blockDim.x + threadIdx.x;
	int idny = blockIdx.y*blockDim.y + threadIdx.y;

	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd   = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep  = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = wB * BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep  = BLOCK_SIZE;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep) {

			// Declaration of the shared memory array As used to
			// store the sub-matrix of A
			__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

			// Declaration of the shared memory array Bs used to
			// store the sub-matrix of B
			__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

			// Load the matrices from device memory
			// to shared memory; each thread loads
			// one element of each matrix
			AS(ty, tx) = A[a + wA * ty + tx];
			BS(ty, tx) = B[b + wB * ty + tx];

			// Synchronize to make sure the matrices are loaded
			__syncthreads();

			// Calculate the distance between the two matrices;
			// each thread computes one element
			// of the block sub-matrix
			// for (int k = 0; k < BLOCK_SIZE && (a+k) <= aEnd; ++k)
			//     Csub += pow(AS(ty, k) - BS(tx, k),2);

			if((a+0) <= aEnd) Csub += pow(AS(ty, 0) - BS(tx, 0),2);
			if((a+1) <= aEnd) Csub += pow(AS(ty, 1) - BS(tx, 1),2);
			if((a+2) <= aEnd) Csub += pow(AS(ty, 2) - BS(tx, 2),2);
			if((a+3) <= aEnd) Csub += pow(AS(ty, 3) - BS(tx, 3),2);
			if((a+4) <= aEnd) Csub += pow(AS(ty, 4) - BS(tx, 4),2);
			if((a+5) <= aEnd) Csub += pow(AS(ty, 5) - BS(tx, 5),2);
			if((a+6) <= aEnd) Csub += pow(AS(ty, 6) - BS(tx, 6),2);
			if((a+7) <= aEnd) Csub += pow(AS(ty, 7) - BS(tx, 7),2);
			if((a+8) <= aEnd) Csub += pow(AS(ty, 8) - BS(tx, 8),2);
			if((a+9) <= aEnd) Csub += pow(AS(ty, 9) - BS(tx, 9),2);
			if((a+10) <= aEnd) Csub += pow(AS(ty, 10) - BS(tx, 10),2);
			if((a+11) <= aEnd) Csub += pow(AS(ty, 11) - BS(tx, 11),2);
			if((a+12) <= aEnd) Csub += pow(AS(ty, 12) - BS(tx, 12),2);
			if((a+13) <= aEnd) Csub += pow(AS(ty, 13) - BS(tx, 13),2);
			if((a+14) <= aEnd) Csub += pow(AS(ty, 14) - BS(tx, 14),2);
			if((a+15) <= aEnd) Csub += pow(AS(ty, 15) - BS(tx, 15),2);

			// Synchronize to make sure that the preceding
			// computation is done before loading two new
			// sub-matrices of A and B in the next iteration
			__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	if(idnx < wC && idny < hC){
		int c = hC * BLOCK_SIZE * bx + BLOCK_SIZE * by;
		C[c + hC * tx + ty] = exp(-(Csub/(scalingfactor*pow(c_width[idnx],2))));         
	}
}


extern "C" void KernelActivationMatrix(cudafloat *d_C, cudafloat* d_A, cudafloat* d_B,int uiWA,int uiWB, int uiWC, int uiHC, float scalingfactor, float* c_width)
{

	int blockSize = 16;

	int wBlocks = uiWC/blockSize + ((uiWC%blockSize == 0)?0:1);
	int hBlocks = uiHC/blockSize + ((uiHC%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);

	ActivationMatrix<<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB, uiWC, uiHC,scalingfactor,c_width);

}




