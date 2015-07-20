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
#include "rankernels.h"

namespace GPUMLib {

/* KERNEL Euclidian Distance */
KERNEL EuclidianDistance(cudafloat *Output, int output_height, int output_width, cudafloat *Input, int input_width, cudafloat *Centers, int centers_width){

	int idnx = blockIdx.x*blockDim.x + threadIdx.x;
	int idny = blockIdx.y*blockDim.y + threadIdx.y;

	if(idnx < output_width && idny < output_height){

		float sum = 0;

		float a;
		float b;

		for(int i = 0; i < centers_width; i++){

			a = Centers[idnx * centers_width + i];
			b = Input[idny * input_width + i];

			sum = sum + pow( a - b , 2);

		}

		Output[idnx + idny * output_width] = sqrt(sum);
	}
}

extern "C" void KernelEuclidianDistance(cudafloat *Output, int output_height, int output_width, cudafloat *Input, int input_width, cudafloat *Centers, int centers_width)
{
	int blockSize = 16;

	int wBlocks = output_width/blockSize + ((output_width%blockSize == 0)?0:1);
	int hBlocks = output_height/blockSize + ((output_height%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	EuclidianDistance<<<grid,threads>>>(Output, output_height, output_width, Input, input_width, Centers, centers_width);
}

KERNEL FindMinKernel(cudafloat *Output, int output_height, int output_width, float *min_array, int* min_idx, cudafloat* Targets){

	int idny = blockIdx.y*blockDim.y + threadIdx.y;

	float min_tmp = -1;
	int idx = 0;

	if(idny < output_height){

		for(int j = 0; j < output_width; j++){
			if(Targets[j] != Targets[idny] && (Output[idny*output_width+j] < min_tmp || min_tmp == -1)){
				min_tmp = Output[idny*output_width+j];
				idx = j;
			}
		}

		min_array[idny] = min_tmp;
		min_idx[idny] = idx;
	}
}

extern "C" void FindMin(cudafloat *Output, int output_height, int output_width, float *min_array, int* min_idx, cudafloat* Targets)
{
	int blockSize = 16;

	int wBlocks = 1;
	int hBlocks = output_height/blockSize + ((output_height%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	FindMinKernel<<<grid,threads>>>(Output, output_height, output_width, min_array, min_idx, Targets);
}



/********************/
/* KERNEL Euclidian Distance */
KERNEL FindNearestCenter(cudafloat *Output, int output_width, cudafloat *Sample, cudafloat *Centers, int centers_width, float* min_value){

	int idnx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idnx < output_width){

		double sum = 0;

		double a;
		double b;

		for(int i = 0; i < centers_width; i++){

			a = Centers[idnx * centers_width + i];
			b = Sample[i];

			sum = sum + pow( a - b , 2);

		}

		Output[idnx] = sqrt(sum);
	}

	__syncthreads();

	if(idnx == 0){

		min_value[0] = Output[0];

		for(int i = 0; i < output_width; i++){

			if(min_value[0] > Output[i]){
				min_value[0] = Output[i];
			}

		}

	}



}

extern "C" void KernelFindNearestCenter(cudafloat *Output, int output_width, cudafloat *Sample, cudafloat *Centers, int centers_width, float* min_value)
{
	int blockSize = 16;

	int wBlocks = output_width/blockSize + ((output_width%blockSize == 0)?0:1);
	int hBlocks = 1;//output_height/blockSize + ((output_height%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	FindNearestCenter<<<grid,threads>>>(Output, output_width, Sample, Centers, centers_width,min_value);
}


/****************/
KERNEL ActivationMatrix(cudafloat *Output, int output_height, int output_width, cudafloat *Input, int input_width, cudafloat *Centers, int centers_width, float* c_width, float scalingfactor){

	int idnx = blockIdx.x*blockDim.x + threadIdx.x;
	int idny = blockIdx.y*blockDim.y + threadIdx.y;

	if(idnx < output_width && idny < output_height){

		float sum = 0;

		float a;
		float b;

		for(int i = 0; i < centers_width; i++){

			a = Centers[idnx * centers_width + i];
			b = Input[idny * input_width + i];

			sum = sum + pow( a - b , 2);

		}

		sum = sqrt(sum);

		//column-major
		float value = exp(-(pow(sum,2)/(scalingfactor*pow(c_width[idnx],2))));

		if(IsInfOrNaN(value)) value = CUDA_VALUE(0.0);

		Output[idnx * output_height + idny] = value;

		//row-major
		//Output[idnx + output_width * idny] = exp(-(pow(sum,2)/(scalingfactor*pow(c_width[idnx],2))));

	}
}

extern "C" void KernelActivationMatrix(cudafloat *Output, int output_height, int output_width, cudafloat *Input, int input_width, cudafloat *Centers, int centers_width, float *c_width, float scalingfactor)
{
	int blockSize = 16;

	int wBlocks = output_width/blockSize + ((output_width%blockSize == 0)?0:1);
	int hBlocks = output_height/blockSize + ((output_height%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	ActivationMatrix<<<grid,threads>>>(Output, output_height, output_width, Input, input_width, Centers, centers_width,c_width,scalingfactor);
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

/************************/
KERNEL CalculateDistance(cudafloat* Output, cudafloat* A, cudafloat* B, unsigned int n){
	extern __shared__ cudafloat sdata[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

	sdata[tid] = (i < n) ? pow(A[i]-B[i],2) : 0;
	if (i + blockDim.x < n) 
		sdata[tid] += pow(A[i+blockDim.x]-B[i+blockDim.x],2);

	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
	{
		if (tid < s) 
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem 
	if(tid == 0) Output[blockIdx.x] = sdata[0];
}


extern "C" unsigned int nextPow2( unsigned int x ) {
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

extern "C" cudafloat KernelCalculateDistance(cudafloat *output, cudafloat *A, cudafloat *B,int n)
{
	int blockSize = 16;
	bool needReadBack = true;

	int threads = (n < blockSize*2) ? nextPow2((n + 1)/ 2) : blockSize;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(cudafloat) : threads * sizeof(cudafloat);

	dim3 dimGrid(blocks, 1, 1);
	dim3 dimBlock(threads, 1, 1);

	CalculateDistance<<<dimGrid,dimBlock,smemSize>>>(output,A,B,n);

	int s = blocks;

	while(s > 15){	   
		threads = (s < blockSize*2) ? nextPow2((s + 1)/ 2) : blockSize;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);
		smemSize = (threads <= 32) ? 2 * threads * sizeof(cudafloat) : threads * sizeof(cudafloat);

		dim3 dimGrid1(blocks, 1, 1);
		dim3 dimBlock1(threads, 1, 1);

		CalculateDistance<<<dimGrid1,dimBlock1,smemSize>>>(output,A,B,s);

		s = (s + (threads * 2 - 1)) / (threads * 2);
	}

	cudafloat gpu_result = 0;

	if (s > 1)
	{
		// copy result from device to host

		cudafloat* h_odata = (cudafloat*) malloc(sizeof(cudafloat)*s);

		cudaMemcpy( h_odata, output, s * sizeof(cudafloat), cudaMemcpyDeviceToHost);

		for(int i=0; i < s; i++) 
		{
			gpu_result += h_odata[i];
		}

		needReadBack = false;

		free(h_odata);

	}

	if (needReadBack)
	{
		cudaMemcpy( &gpu_result, output, sizeof(cudafloat), cudaMemcpyDeviceToHost);
	}

	return sqrt(gpu_result);
}

















#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]

#define BLOCK_SIZE 16

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
__global__ void
matrixMul( float* C, float* A, float* B, int wA, int wB, int wC, int hC)
{

	int idnx = blockIdx.x*blockDim.x + threadIdx.x;
	int idny = blockIdx.y*blockDim.y + threadIdx.y;

	if(idnx < wC && idny < hC){

		float sum = 0;

		float a;
		float b;

		for(int i = 0; i < wA; i++){

			a = A[i * hC + idny];

			b = B[i * wC + idnx];

			sum = sum + (a*b);

		}

		//row-major
		C[idnx + wC * idny] = sum;

	}


}

extern "C" void matmul(cudafloat *d_C, cudafloat* d_A, cudafloat* d_B,int uiWA,int uiWB, int uiWC, int uiHC)
{
	int blockSize = 16;

	int wBlocks = uiWC/blockSize + ((uiWC%blockSize == 0)?0:1);
	int hBlocks = uiHC/blockSize + ((uiHC%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);

	matrixMul<<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB, uiWC, uiHC);

}












KERNEL CalculateNetworkActivation(cudafloat* output, cudafloat* Sample,int Length,cudafloat* dCenters,int NumCenters,cudafloat* dWeights,int NumClasses,cudafloat* dWidths,float scaling_factor){

	int idnx = blockIdx.x*blockDim.x + threadIdx.x;
	int idny = blockIdx.y*blockDim.y + threadIdx.y;

	if(idnx < NumClasses && idny < NumCenters){

		float sum = 0;

		float a;
		float b;

		for(int i = 0; i < Length; i++){

			a = dCenters[idny * Length + i];
			b = Sample[i];

			sum = sum + pow( a - b , 2);

		}

		sum = sqrt(sum);

		//column-major
		//Output[idnx * output_height + idny] = exp(-(pow(sum,2)/(scalingfactor*pow(c_width[idnx],2))));

		//row-major
		output[idnx + NumClasses * idny] = dWeights[idny*NumClasses+idnx]*exp(-(pow(sum,2)/(scaling_factor*pow(dWidths[idny],2))));

	}

}

extern "C" void KernelCalculateNetworkActivation(cudafloat* output, cudafloat* Sample,int Length,cudafloat* dCenters,int NumCenters,cudafloat* dWeights,int NumClasses,cudafloat* dWidths,float scaling_factor)
{
	int blockSize = 16;

	int wBlocks = NumClasses/blockSize + ((NumClasses%blockSize == 0)?0:1);
	int hBlocks = NumCenters/blockSize + ((NumCenters%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	CalculateNetworkActivation<<<grid,threads>>>(output,Sample,Length,dCenters,NumCenters,dWeights,NumClasses,dWidths,scaling_factor);
}




KERNEL UpdateWidths(cudafloat* dWidths, cudafloat* newWidths, int Length){

	int idnx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idnx < Length){

		if(dWidths[idnx] > newWidths[idnx]){
			dWidths[idnx] = newWidths[idnx];
		}

	}
}

extern "C" void KernelUpdateWidths(cudafloat* dWidths, cudafloat* newWidths, int Length)
{
	int blockSize = 16;

	int wBlocks = Length/blockSize + ((Length%blockSize == 0)?0:1);
	int hBlocks = 1;

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	UpdateWidths<<<grid,threads>>>(dWidths,newWidths,Length);
}





KERNEL CalculateError(cudafloat* result, cudafloat* target, cudafloat* output, int Length){

	int idnx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idnx < Length){

		output[idnx] = sqrt(pow(result[idnx]-target[idnx],2));

		if(IsInfOrNaN(output[idnx])){
			output[idnx] = 0;
		}

	}

}

KERNEL ReduceError(cudafloat* output,cudafloat* error, int Length){

	float sum = 0;

	for(int i = 0; i < Length; i++){
		sum += pow(output[i],2);
	}
	error[0] = sqrt(sum/Length);

}

extern "C" void KernelCalculateError(cudafloat* result, cudafloat* target, cudafloat* output, int Length,float* error)
{
	int blockSize = 16;

	int wBlocks = Length/blockSize + ((Length%blockSize == 0)?0:1);
	int hBlocks = 1;

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	CalculateError<<<grid,threads>>>(result,target,output,Length);

	dim3 grid2(1,1);
	dim3 threads2(1,1);
	ReduceError<<<grid2,threads2>>>(output,error,Length);
}




KERNEL SumActivations(cudafloat* output, int Length, int NumCenters){

	int idnx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idnx < Length){

		float sum = 0;

		for(int i = 0; i < NumCenters; i++){
			sum += output[idnx+Length*i]; 
		}

		output[idnx] = sum;
	}

}

extern "C" void KernelSumActivations(cudafloat* output, int Length, int NumCenters)
{
	int blockSize = 16;

	int wBlocks = Length/blockSize + ((Length%blockSize == 0)?0:1);
	int hBlocks = 1;

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	SumActivations<<<grid,threads>>>(output,Length,NumCenters);
}





KERNEL Copy(cudafloat* dst, cudafloat *src,int Length){

	int idnx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idnx < Length){

		dst[idnx] = src[idnx];

	}

}

extern "C" void KernelCopyTo(cudafloat* dst, cudafloat *src,int Length)
{
	int blockSize = 16;

	int wBlocks = Length/blockSize + ((Length%blockSize == 0)?0:1);
	int hBlocks = 1;

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	Copy<<<grid,threads>>>(dst,src,Length);
}

}
