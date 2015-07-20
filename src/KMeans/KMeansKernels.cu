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
#include "kmeanskernels.h"





#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]

#define BLOCK_SIZE 16

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
__global__ void
EuclidianDistance( float* C, float* A, float* B, int wA, int wB, int wC, int hC)
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
			if((a + wA * ty + tx) < hC*wA)
				AS(ty, tx) = A[a + wA * ty + tx];
			if((b + wB * ty + tx) < wC*wB)
				BS(ty, tx) = B[b + wB * ty + tx];

			// Synchronize to make sure the matrices are loaded
			__syncthreads();

			// Multiply the two matrices together;
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

void KernelEuclidianDistance(cudafloat *d_C, cudafloat* d_A, cudafloat* d_B,int uiWA,int uiWB, int uiWC, int uiHC)
{

	int blockSize = 16;

	int wBlocks = uiWC/blockSize + ((uiWC%blockSize == 0)?0:1);
	int hBlocks = uiHC/blockSize + ((uiHC%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);

	EuclidianDistance<<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB, uiWC, uiHC);

}





KERNEL CenterAttribution(cudafloat *Output, int output_height, int output_width, int *attrib_center){

	int idnx = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	int idny = __umul24(blockIdx.y,blockDim.y) + threadIdx.y;

	float min_tmp = -1;
	int idx = 0;

	if(idny < output_height){

		for(int j = 0; j < output_width; j++){
			if(Output[__umul24(idny,output_width)+j] < min_tmp || min_tmp == -1){
				min_tmp = Output[__umul24(idny,output_width)+j];
				idx = j;
			}
		}

		attrib_center[idny] = idx;
	}
}

void KernelCenterAttribution(cudafloat *Output, int output_height, int output_width, int *attrib_center)
{
	int blockSize = 16;

	int wBlocks = 1;
	int hBlocks = output_height/blockSize + ((output_height%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	CenterAttribution<<<grid,threads>>>(Output, output_height, output_width, attrib_center);
}



KERNEL PrepareCenterCopy(cudafloat *Output, int output_height, int output_width, int *attrib_center){

	int idnx = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	int idny = __umul24(blockIdx.y,blockDim.y) + threadIdx.y;

	int blocks = 1;

	if(idnx < output_width){

		int count = 0;

		int start = blockIdx.y * output_height/blocks;
		int stop = start + output_height/blocks;


		for(int j = start; j < stop; j++){

			if(attrib_center[j] == idnx){

				Output[(start+count)*output_width + idnx] = j;

				count = count + 1;	
			}

		}

		/*   if(attrib_center[stop-15] == idnx){ Output[(start+count)*output_width + idnx] = stop-15; count = count + 1;	}
		if(attrib_center[stop-14] == idnx){ Output[(start+count)*output_width + idnx] = stop-14; count = count + 1;	}
		if(attrib_center[stop-13] == idnx){ Output[(start+count)*output_width + idnx] = stop-13; count = count + 1;	}
		if(attrib_center[stop-12] == idnx){ Output[(start+count)*output_width + idnx] = stop-12; count = count + 1;	}
		if(attrib_center[stop-11] == idnx){ Output[(start+count)*output_width + idnx] = stop-11; count = count + 1;	}
		if(attrib_center[stop-10] == idnx){ Output[(start+count)*output_width + idnx] = stop-10; count = count + 1;	}
		if(attrib_center[stop-9] == idnx){ Output[(start+count)*output_width + idnx] = stop-9; count = count + 1;	}
		if(attrib_center[stop-8] == idnx){ Output[(start+count)*output_width + idnx] = stop-8; count = count + 1;	}
		if(attrib_center[stop-7] == idnx){ Output[(start+count)*output_width + idnx] = stop-7; count = count + 1;	}
		if(attrib_center[stop-6] == idnx){ Output[(start+count)*output_width + idnx] = stop-6; count = count + 1;	}
		if(attrib_center[stop-5] == idnx){ Output[(start+count)*output_width + idnx] = stop-5; count = count + 1;	}
		if(attrib_center[stop-4] == idnx){ Output[(start+count)*output_width + idnx] = stop-4; count = count + 1;	}
		if(attrib_center[stop-3] == idnx){ Output[(start+count)*output_width + idnx] = stop-3; count = count + 1;	}
		if(attrib_center[stop-2] == idnx){ Output[(start+count)*output_width + idnx] = stop-2; count = count + 1;	}
		if(attrib_center[stop-1] == idnx){ Output[(start+count)*output_width + idnx] = stop-1; count = count + 1;	}*/

		__syncthreads();

		attrib_center[idnx+blockIdx.y*output_width] = count;
	}

}

void KernelPrepareCenterCopy(cudafloat *Output, int output_height, int output_width, int *attrib_center)
{


	int blockSize = 16;

	int wBlocks = output_width/blockSize + ((output_width%blockSize == 0)?0:1);
	int hBlocks = 1;   

	dim3 grid(wBlocks,hBlocks,1);
	dim3 threads(blockSize,blockSize);

	PrepareCenterCopy<<<grid,threads>>>(Output,output_height,output_width,attrib_center);


}



KERNEL CopyCenters(cudafloat *Output, int output_height, int output_width, cudafloat *Input,int input_width, int *attrib_center,cudafloat *Indexes, int idx_height, int idx_width){

	int idnx = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	int idny = __umul24(blockIdx.y,blockDim.y) + threadIdx.y;

	int blocks = 1;

	if(idny < output_height && idnx < output_width){

		int idx = __umul24(idny,output_width);

		int count = 0;
		float aux = 0;


		for(int block_number = 0; block_number < blocks; block_number++){

			int start = block_number * idx_height/blocks;

			for(int j = 0; j < attrib_center[idny+block_number*output_height]; j++){

				aux = aux + Input[__umul24(Indexes[(j+start)*idx_width + idny],output_width)+idnx];

				count = count + 1;	

			}
		}

		if(count > 0)
			Output[idx + idnx] = aux/count;
		else
			Output[idx + idnx] = 0;

	}

}

void KernelCopyCenters(cudafloat *Output, int output_height, int output_width, cudafloat *Input,int input_width, int *attrib_center, cudafloat *Indexes, int idx_height, int idx_width)
{


	int blockSize = 16;

	int wBlocks = output_width/blockSize + ((output_width%blockSize == 0)?0:1);;
	int hBlocks = output_height/blockSize + ((output_height%blockSize == 0)?0:1);   

	dim3 grid(wBlocks,hBlocks,1);
	dim3 threads(blockSize,blockSize);

	CopyCenters<<<grid,threads>>>(Output,output_height,output_width,Input,input_width,attrib_center,Indexes,idx_height,idx_width);


}




unsigned int nextPow2( unsigned int x ) {
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

KERNEL reduce2(int *g_idata,int *g_odata, int *g_idata_old, unsigned int n)
{
	extern __shared__ int sdata2[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

	sdata2[tid] = (i < n) ? ((g_idata[i] == g_idata_old[i])?0:1) : 0;
	if (i + blockDim.x < n)
		sdata2[tid] += (g_idata[i+blockDim.x] == g_idata_old[i+blockDim.x])?0:1;
	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
	{
		if (tid < s) 
		{
			sdata2[tid] += sdata2[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem 
	if(tid == 0) g_odata[blockIdx.x] = sdata2[0];
}

KERNEL reduce3(int *g_idata,int *g_odata, unsigned int n)
{
	extern __shared__ int sdata3[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

	sdata3[tid] = (i < n) ? g_idata[i] : 0;
	if (i + blockDim.x < n) 
		sdata3[tid] += g_idata[i+blockDim.x];  

	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
	{
		if (tid < s) 
		{
			sdata3[tid] += sdata3[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem 
	if(tid == 0) g_odata[blockIdx.x] = sdata3[0];
}

void KernelReduce2(int *output, int *input, int *g_idata_old,int n)
{
	int blockSize = 16;

	int threads = (n < blockSize*2) ? nextPow2((n + 1)/ 2) : blockSize;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);

	dim3 dimGrid(blocks, 1, 1);
	dim3 dimBlock(threads, 1, 1);

	reduce2<<<dimGrid,dimBlock,smemSize>>>(input,output,g_idata_old,n);

	int s = blocks;

	while(s > 1){	   
		threads = (s < blockSize*2) ? nextPow2((s + 1)/ 2) : blockSize;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);
		smemSize = (threads <= 32) ? 2 * threads * sizeof(cudafloat) : threads * sizeof(cudafloat);

		dim3 dimGrid1(blocks, 1, 1);
		dim3 dimBlock1(threads, 1, 1);

		reduce3<<<dimGrid1,dimBlock1,smemSize>>>(output,output,s);

		s = (s + (threads * 2 - 1)) / (threads * 2);
	}

	/*   int gpu_result = 0;

	if (s > 1)
	{
	// copy result from device to host

	int* h_odata = (int*) malloc(sizeof(int)*s);

	cudaMemcpy( h_odata, output, s * sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0; i < s; i++) 
	{
	gpu_result += h_odata[i];
	}

	needReadBack = false;

	free(h_odata);

	}

	if (needReadBack)
	{
	//      cudaMemcpy( &gpu_result, output, sizeof(int), cudaMemcpyDeviceToHost);
	}

	//   return gpu_result;*/
}



















/* KERNEL Euclidian Distance */
__device__ float distance( float *v1, float *v2, int dimensions)
{
	float dist = 0;

	for( int i = 0; i < dimensions; i++ )
	{
		float tmp = v2[i] - v1[i];
		dist += tmp * tmp;
	}

	return sqrt(dist);
}

KERNEL CenterAttribution_Bounds(cudafloat *Output, int output_height, int output_width, int *attrib_center, float* upperbound){

	int idny = blockIdx.y*blockDim.y + threadIdx.y;

	float min_tmp = -1;
	int idx = 0;

	if(idny < output_height){

		for(int j = 0; j < output_width; j++){
			if(Output[idny*output_width+j] < min_tmp || min_tmp == -1){
				min_tmp = Output[idny*output_width+j];
				idx = j;
			}
		}

		attrib_center[idny] = idx;
		upperbound[idny] = Output[idny*output_width+idx];
	}
}

void KernelCenterAttribution_Bounds(cudafloat *Output, int output_height, int output_width, int *attrib_center, float* upperbound)
{
	int blockSize = 16;

	int wBlocks = 1;
	int hBlocks = output_height/blockSize + ((output_height%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	CenterAttribution_Bounds<<<grid,threads>>>(Output, output_height, output_width, attrib_center,upperbound);
}





KERNEL CopyCenters2(cudafloat *Output, int output_height, int output_width, cudafloat *Input,int input_width, int *attrib_center){

	int idnx = blockIdx.x*blockDim.x + threadIdx.x;
	int idny = blockIdx.y*blockDim.y + threadIdx.y;

	if(idny < output_height && idnx < output_width){

		int count = 0;
		float aux = 0;

		for(int j = 0; j < input_width; j++){

			if(attrib_center[j] == idny){

				aux = aux + Input[j*output_width+idnx];

				count = count + 1;	
			}

		}

		Output[idny * output_width + idnx] = aux/count;

	}

}

void KernelCopyCenters2(cudafloat *Output, int output_height, int output_width, cudafloat *Input,int input_width, int *attrib_center)
{
	int blockSize = 16;

	int wBlocks = output_width/blockSize + ((output_width%blockSize == 0)?0:1);;
	int hBlocks = output_height/blockSize + ((output_height%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);

	CopyCenters2<<<grid,threads>>>(Output,output_height,output_width,Input,input_width,attrib_center);
}








KERNEL FindS(cudafloat *Output, int output_height, int output_width, float *S){

	int idny = blockIdx.y*blockDim.y + threadIdx.y;

	float min_tmp = -1;


	if(idny < output_height){

		for(int j = 0; j < output_width; j++){
			if(idny != j){
				if(Output[idny*output_width+j] < min_tmp || min_tmp == -1){
					min_tmp = Output[idny*output_width+j];
				}
			}
		}

		S[idny] = CUDA_VALUE(0.5) * min_tmp;
	}
}

void KernelS(cudafloat *Output, int output_height, int output_width, float *S)
{



	int blockSize = 16;

	int wBlocks = 1;
	int hBlocks = output_height/blockSize + ((output_height%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	FindS<<<grid,threads>>>(Output, output_height, output_width, S);

}










KERNEL Step3(float*Input,int input_height, float* UpperBounds, float* S, bool* R,int* CenterAttrib,float* LowerBounds,float* DistanceBeetweenCenters,float* InitialDistances, float* NewCenters,int centers_height,int centers_width){

	int idny = blockIdx.y*blockDim.y + threadIdx.y;

	if(idny < input_height){

		float upperbound = UpperBounds[idny];
		int centeratt = CenterAttrib[idny];
		float s_value = S[centeratt];

		if(!(upperbound <= s_value)){

			for(int j = 0; j < centers_height; j++){

				if(j != centeratt && upperbound > LowerBounds[idny*centers_height+j] && upperbound > CUDA_VALUE(0.5)*DistanceBeetweenCenters[centeratt*centers_height+j]){

					//std::cout << " - Step 3a - " << std::endl;
					if(R[idny]){

						InitialDistances[idny*centers_height+centeratt] = distance(&(Input[idny*centers_width]),&(NewCenters[centeratt*centers_width]),centers_width);

						upperbound = InitialDistances[idny*centers_height+centeratt];

						R[idny] = false;

					}else{

						InitialDistances[idny*centers_height+centeratt] = upperbound;

					}
				}

				//std::cout << " - Step 3b - " << std::endl;
				if(InitialDistances[idny*centers_height+centeratt] > LowerBounds[idny*centers_height+j] ||
					InitialDistances[idny*centers_height+centeratt] > CUDA_VALUE(0.5) * DistanceBeetweenCenters[centeratt*centers_height+j]){
						//std::cout << " - Step 3b - " << std::endl;

						InitialDistances[idny*centers_height+j] = distance(&(Input[idny*centers_width]),&(NewCenters[j*centers_width]),centers_width);

						LowerBounds[idny*centers_height+j] = InitialDistances[idny*centers_height+j];

						if(InitialDistances[idny*centers_height+j] < InitialDistances[idny*centers_height+centeratt]){

							centeratt = j;

							upperbound = InitialDistances[idny*centers_height+centeratt];

						}

				}

			}

		}

		UpperBounds[idny] = upperbound;
		CenterAttrib[idny] = centeratt;

	}

}

void KernelStep3(float* Input,int input_height, float* Upperbounds, float* S,bool* R,int* CenterAttrib,float* LowerBounds,float* DistanceBeetweenCenters,float* InitialDistances, float* NewCenters,int centers_height,int centers_width)
{


	int blockSize = 16;

	int wBlocks = 1;
	int hBlocks = input_height/blockSize + ((input_height%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	Step3<<<grid,threads>>>(Input,input_height,Upperbounds,S,R,CenterAttrib,LowerBounds,DistanceBeetweenCenters,InitialDistances,NewCenters,centers_height,centers_width);

}



KERNEL Step5(int input_height, float* UpperBounds, bool* R,int* CenterAttrib,float* LowerBounds,float* DistanceBeetweenCenters,float* InitialDistances, float* NewCenters,int centers_height,int centers_width){

	int idny = blockIdx.y*blockDim.y + threadIdx.y;

	if(idny < input_height){

		for(int j = 0; j < centers_height; j++){

			if(LowerBounds[idny*centers_height+j] - DistanceBeetweenCenters[j*centers_height+j] > 0){
				LowerBounds[idny*centers_height+j] = LowerBounds[idny*centers_height+j] - DistanceBeetweenCenters[j*centers_height+j];
			}else{
				LowerBounds[idny*centers_height+j] = 0;
			}
		}

		UpperBounds[idny] = UpperBounds[idny] + DistanceBeetweenCenters[CenterAttrib[idny]*centers_height+CenterAttrib[idny]];

		R[idny] = true;

	}

}

void KernelStep5(int input_height, float* Upperbounds, bool* R,int* CenterAttrib,float* LowerBounds,float* DistanceBeetweenCenters,float* InitialDistances, float* NewCenters,int centers_height,int centers_width)
{


	int blockSize = 16;

	int wBlocks = 1;
	int hBlocks = input_height/blockSize + ((input_height%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	Step5<<<grid,threads>>>(input_height,Upperbounds,R,CenterAttrib,LowerBounds,DistanceBeetweenCenters,InitialDistances,NewCenters,centers_height,centers_width);


}















KERNEL reduce_bool(bool *g_idata,bool *g_odata, unsigned int n)
{
	extern __shared__ bool sdatabool[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

	sdatabool[tid] = (i < n) ? g_idata[i] : true;
	if (i + blockDim.x < n) 
		sdatabool[tid] = sdatabool[tid] && g_idata[i+blockDim.x];  

	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
	{
		if (tid < s) 
		{
			sdatabool[tid] = sdatabool[tid] && sdatabool[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem 
	if(tid == 0) g_odata[blockIdx.x] = sdatabool[0];
}

void KernelReduce_bool(bool *output, bool *input,int n)
{
	int blockSize = 16;

	int threads = (n < blockSize*2) ? nextPow2((n + 1)/ 2) : blockSize;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(cudafloat) : threads * sizeof(cudafloat);

	dim3 dimGrid(blocks, 1, 1);
	dim3 dimBlock(threads, 1, 1);

	reduce_bool<<<dimGrid,dimBlock,smemSize>>>(input,output,n);

	int s = blocks;

	while(s > 1){	   
		threads = (s < blockSize*2) ? nextPow2((s + 1)/ 2) : blockSize;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);
		smemSize = (threads <= 32) ? 2 * threads * sizeof(cudafloat) : threads * sizeof(cudafloat);

		dim3 dimGrid1(blocks, 1, 1);
		dim3 dimBlock1(threads, 1, 1);

		reduce_bool<<<dimGrid1,dimBlock1,smemSize>>>(output,output,s);

		s = (s + (threads * 2 - 1)) / (threads * 2);
	}

	/* bool gpu_result = true;

	if (s > 1)
	{
	// copy result from device to host

	bool* h_odata = (bool*) malloc(sizeof(bool)*s);

	cudaMemcpy( h_odata, output, s * sizeof(bool), cudaMemcpyDeviceToHost);

	for(int i=0; i < s; i++) 
	{

	gpu_result = gpu_result && h_odata[i];
	}


	needReadBack = false;

	free(h_odata);

	}

	if (needReadBack)
	{
	cudaMemcpy( &gpu_result, output, sizeof(bool), cudaMemcpyDeviceToHost);
	}

	return gpu_result;*/
}