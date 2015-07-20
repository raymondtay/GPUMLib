/*
	Noel Lopes is an Assistant Professor at the Polytechnic Institute of Guarda, Portugal
	Copyright (C) 2009, 2010, 2011, 2012 Noel de Jesus Mendonça Lopes

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

#include "SumWarp.h"
#include "reduction.h"

namespace GPUMLib {

template <int blockSize> KERNEL Sum(cudafloat * inputs, cudafloat * outputs, int numInputs) {
	extern __shared__ cudafloat sum[];
  
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cudafloat value = CUDA_VALUE(0.0);
	if (idx < numInputs) value = inputs[idx];

	sum[threadIdx.x] = value;
	__syncthreads();

	SumBeforeWarp<blockSize>(sum);

	if (threadIdx.x < 32) {
		SumWarp<blockSize>(sum);
		if (threadIdx.x == 0) outputs[blockIdx.x] = sum[0];
	}
}

void KernelSum(cudaStream_t stream, int blocks, int blockSize, cudafloat * inputs, cudafloat * outputs, int numInputs) {
	switch(blockSize) {
		#ifdef FERMI
		case 1024:
			Sum<1024><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, outputs, numInputs);
			break;
		#endif
		case 512:
			Sum<512><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, outputs, numInputs);
			break;
		case 256:
			Sum<256><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, outputs, numInputs);
			break;
		case 128:
			Sum<128><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, outputs, numInputs);
			break;
		case 64:
			Sum<64><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, outputs, numInputs);
			break;
		case 32:
			Sum<32><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, outputs, numInputs);
			break;
		case 16:
			Sum<16><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, outputs, numInputs);
			break;
		case 8:
			Sum<8><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, outputs, numInputs);
			break;
		case 4:
			Sum<4><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, outputs, numInputs);
			break;
		case 2:
			Sum<2><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, outputs, numInputs);
			break;
		case 1:
			Sum<1><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, outputs, numInputs);
			break;
	}
}

template <int blockSize> KERNEL SumSmallArray(cudafloat * inputs, cudafloat * output, int numInputs, cudafloat multiplyFactor) {
	extern __shared__ cudafloat sum[];
  
	cudafloat value = CUDA_VALUE(0.0);
	for(int i = threadIdx.x; i < numInputs; i += blockDim.x) value += inputs[i]; 
	sum[threadIdx.x] = value;
	__syncthreads();

	SumBeforeWarp<blockSize>(sum);

	if (threadIdx.x < 32) {
		SumWarp<blockSize>(sum);

		if (threadIdx.x == 0) output[blockIdx.x] = sum[0] * multiplyFactor;
	}
}

void KernelSumSmallArray(cudaStream_t stream, int blockSize, cudafloat * inputs, cudafloat * output, int numInputs, cudafloat multiplyFactor) {
	switch(blockSize) {
		#ifdef FERMI
		case 1024:
			SumSmallArray<1024><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs, multiplyFactor);
			break;
		#endif
		case 512:
			SumSmallArray<512><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs, multiplyFactor);
			break;
		case 256:
			SumSmallArray<256><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs, multiplyFactor);
			break;
		case 128:
			SumSmallArray<128><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs, multiplyFactor);
			break;
		case 64:
			SumSmallArray<64><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs, multiplyFactor);
			break;
		case 32:
			SumSmallArray<32><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs, multiplyFactor);
			break;
		case 16:
			SumSmallArray<16><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs, multiplyFactor);
			break;
		case 8:
			SumSmallArray<8><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs, multiplyFactor);
			break;
		case 4:
			SumSmallArray<4><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs, multiplyFactor);
			break;
		case 2:
			SumSmallArray<2><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs, multiplyFactor);
			break;
		case 1:
			SumSmallArray<1><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs, multiplyFactor);
			break;
	}
}

}