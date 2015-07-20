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

#include "../common/config.h"
#include "../reduction/SumWarp.h"

namespace GPUMLib {

#define NEURON blockIdx.x
#define NUM_NEURONS gridDim.x

#define SAMPLE blockIdx.y

template <int blockSize> KERNEL ComputeStatusHiddenUnitsRBM(cudafloat * v, cudafloat * weights, cudafloat * b, cudafloat * h, float * randomValues, int I) {
	extern __shared__ cudafloat iw[];
  
	iw[threadIdx.x] = CUDA_VALUE(0.0);
	for(int i = threadIdx.x; i < I; i += blockDim.x) {
		iw[threadIdx.x] += v[SAMPLE * I + i] * weights[NEURON * I + i];
	}
	__syncthreads();

	if (blockSize >= 1024) {
		if (threadIdx.x < 512) iw[threadIdx.x] += iw[threadIdx.x + 512];
		__syncthreads();
	}
	
	if (blockSize >= 512) {
		if (threadIdx.x < 256) iw[threadIdx.x] += iw[threadIdx.x + 256];
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threadIdx.x < 128) iw[threadIdx.x] += iw[threadIdx.x + 128];
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadIdx.x < 64) iw[threadIdx.x] += iw[threadIdx.x + 64];
		__syncthreads();
	}

	__shared__ cudafloat output;
	if (threadIdx.x < 32) {
		SumWarp<blockSize>(iw);
	
		if (threadIdx.x == 0) {
			output = CUDA_SIGMOID(iw[0] + b[NEURON]);
			int idx = SAMPLE * NUM_NEURONS + NEURON;
			if (randomValues != nullptr) output = (output > randomValues[idx]) ? CUDA_VALUE(1.0) : CUDA_VALUE(0.0);
			h[idx] = output;
		}
	}
}

void KernelComputeStatusHiddenUnitsRBM(dim3 & gridSize, int blockSize, cudafloat * v, cudafloat * weights, cudafloat * b, cudafloat * h, float * randomValues, int I) {
	switch(blockSize) {
		#ifdef FERMI
		case 1024:
			ComputeStatusHiddenUnitsRBM<1024><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(v, weights, b,  h, randomValues, I);
			break;
		#endif
		case 512:
			ComputeStatusHiddenUnitsRBM<512><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(v, weights, b,  h, randomValues, I);
			break;
		case 256:
			ComputeStatusHiddenUnitsRBM<256><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(v, weights, b,  h, randomValues, I);
			break;
		case 128:
			ComputeStatusHiddenUnitsRBM<128><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(v, weights, b,  h, randomValues, I);
			break;
		case 64:
			ComputeStatusHiddenUnitsRBM<64><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(v, weights, b,  h, randomValues, I);
			break;
		case 32:
			ComputeStatusHiddenUnitsRBM<32><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(v, weights, b,  h, randomValues, I);
			break;
		case 16:
			ComputeStatusHiddenUnitsRBM<16><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(v, weights, b,  h, randomValues, I);
			break;
		case 8:
			ComputeStatusHiddenUnitsRBM<8><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(v, weights, b,  h, randomValues, I);
			break;
		case 4:
			ComputeStatusHiddenUnitsRBM<4><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(v, weights, b,  h, randomValues, I);
			break;
		case 2:
			ComputeStatusHiddenUnitsRBM<2><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(v, weights, b,  h, randomValues, I);
			break;
		case 1:
			ComputeStatusHiddenUnitsRBM<1><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(v, weights, b,  h, randomValues, I);
			break;
	}
}

template <int blockSize> KERNEL ComputeStatusVisibleUnitsRBM(cudafloat * h, cudafloat * weights, cudafloat * a, cudafloat * v, float * randomValues, int J) {
	extern __shared__ cudafloat sum[];
  
	sum[threadIdx.x] = CUDA_VALUE(0.0);
	for(int j = threadIdx.x; j < J; j += blockDim.x) {
		sum[threadIdx.x] += h[SAMPLE * J + j] * weights[j * NUM_NEURONS + NEURON];
	}
	__syncthreads();

	if (blockSize >= 1024) {
		if (threadIdx.x < 512) sum[threadIdx.x] += sum[threadIdx.x + 512];
		__syncthreads();
	}
	
	if (blockSize >= 512) {
		if (threadIdx.x < 256) sum[threadIdx.x] += sum[threadIdx.x + 256];
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threadIdx.x < 128) sum[threadIdx.x] += sum[threadIdx.x + 128];
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadIdx.x < 64) sum[threadIdx.x] += sum[threadIdx.x + 64];
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		SumWarp<blockSize>(sum);

		if (threadIdx.x == 0) {
			cudafloat output = CUDA_SIGMOID(sum[0] + a[NEURON]);

			int idx = SAMPLE * NUM_NEURONS + NEURON;
			if (randomValues != nullptr) output = (output > randomValues[idx]) ? CUDA_VALUE(1.0) : CUDA_VALUE(0.0);
			v[idx] = output;
		}
	}
}

void KernelComputeStatusVisibleUnitsRBM(dim3 & gridSize, int blockSize, cudafloat * h, cudafloat * weights, cudafloat * a, cudafloat * v, float * randomValues, int J) {
	switch(blockSize) {
		#ifdef FERMI
		case 1024:
			ComputeStatusVisibleUnitsRBM<1024><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(h, weights, a, v, randomValues, J);
			break;
		#endif
		case 512:
			ComputeStatusVisibleUnitsRBM<512><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(h, weights, a, v, randomValues, J);
			break;
		case 256:
			ComputeStatusVisibleUnitsRBM<256><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(h, weights, a, v, randomValues, J);
			break;
		case 128:
			ComputeStatusVisibleUnitsRBM<128><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(h, weights, a, v, randomValues, J);
			break;
		case 64:
			ComputeStatusVisibleUnitsRBM<64><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(h, weights, a, v, randomValues, J);
			break;
		case 32:
			ComputeStatusVisibleUnitsRBM<32><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(h, weights, a, v, randomValues, J);
			break;
		case 16:
			ComputeStatusVisibleUnitsRBM<16><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(h, weights, a, v, randomValues, J);
			break;
		case 8:
			ComputeStatusVisibleUnitsRBM<8><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(h, weights, a, v, randomValues, J);
			break;
		case 4:
			ComputeStatusVisibleUnitsRBM<4><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(h, weights, a, v, randomValues, J);
			break;
		case 2:
			ComputeStatusVisibleUnitsRBM<2><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(h, weights, a, v, randomValues, J);
			break;
		case 1:
			ComputeStatusVisibleUnitsRBM<1><<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(h, weights, a, v, randomValues, J);
			break;
	}
}

}
