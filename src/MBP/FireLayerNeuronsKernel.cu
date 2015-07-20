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

#include "../reduction/SumWarp.h"
#include "MBPkernels.h"

#define BIAS 0

#define NEURON blockIdx.x
#define NUM_NEURONS gridDim.x

#define PATTERN blockIdx.y

namespace GPUMLib {

template <int blockSize> KERNEL FireLayerNeurons(cudafloat * inputs, cudafloat * weights, cudafloat * m, int mOffset, int totalNeuronsWithSelectiveActivation, cudafloat * outputs, int numInputs) {
	extern __shared__ cudafloat iw[];

	iw[threadIdx.x] = CUDA_VALUE(0.0);
	for(int i = threadIdx.x; i <= numInputs; i += blockDim.x) {
		cudafloat i_w = weights[NEURON * (numInputs + 1) + i];
		if (i > BIAS) i_w *= inputs[PATTERN * numInputs + (i - 1)];  
		iw[threadIdx.x] += i_w;
	}
	__syncthreads();

	SumBeforeWarp<blockSize>(iw);

	if (threadIdx.x < 32) {
		SumWarp<blockSize>(iw);
	
		if (threadIdx.x == 0) {
			cudafloat output = CUDA_SIGMOID(iw[0]);
			if (m != nullptr) output *= m[PATTERN * totalNeuronsWithSelectiveActivation + NEURON + mOffset];
			outputs[PATTERN * NUM_NEURONS + NEURON] = output;
		}
	}
}

void KernelFireLayer(cudaStream_t stream, dim3 & gridSize, int blockSize, cudafloat * inputs, cudafloat * weights, cudafloat * m, int mOffset, int totalNeuronsWithSelectiveActivation, cudafloat * outputs, int numInputs) {
	switch(blockSize) {
		#ifdef FERMI
		case 1024:
			FireLayerNeurons<1024><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, outputs, numInputs);
			break;
		#endif
		case 512:
			FireLayerNeurons<512><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, outputs, numInputs);
			break;
		case 256:
			FireLayerNeurons<256><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, outputs, numInputs);
			break;
		case 128:
			FireLayerNeurons<128><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, outputs, numInputs);
			break;
		case 64:
			FireLayerNeurons<64><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, outputs, numInputs);
			break;
		case 32:
			FireLayerNeurons<32><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, outputs, numInputs);
			break;
		case 16:
			FireLayerNeurons<16><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, outputs, numInputs);
			break;
		case 8:
			FireLayerNeurons<8><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, outputs, numInputs);
			break;
		case 4:
			FireLayerNeurons<4><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, outputs, numInputs);
			break;
		case 2:
			FireLayerNeurons<2><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, outputs, numInputs);
			break;
		case 1:
			FireLayerNeurons<1><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, outputs, numInputs);
			break;
	}
}

template <int blockSize> KERNEL FireOutputLayerNeurons(cudafloat * inputs, cudafloat * weights, cudafloat * m, int mOffset, int totalNeuronsWithSelectiveActivation, cudafloat * desiredOutputs, cudafloat * outputs, cudafloat * localGradient, cudafloat * rms, cudafloat * localGradientSpaceNet, int numInputs) {
	extern __shared__ cudafloat iw[];
	
	iw[threadIdx.x] = CUDA_VALUE(0.0);
	for(int i = threadIdx.x; i <= numInputs; i += blockDim.x) {
		cudafloat i_w = weights[NEURON * (numInputs + 1) + i];
		if (i > BIAS) i_w *= inputs[PATTERN * numInputs + (i - 1)];
		iw[threadIdx.x] += i_w;
	}
	__syncthreads();

	SumBeforeWarp<blockSize>(iw);

	if (threadIdx.x < 32) {
		SumWarp<blockSize>(iw);
		
		if (threadIdx.x == 0) {
			int n = PATTERN * NUM_NEURONS + NEURON;
			int nSelAct = PATTERN * totalNeuronsWithSelectiveActivation + NEURON + mOffset;

			cudafloat output = CUDA_SIGMOID(iw[0]);
			cudafloat M = (m != nullptr) ? m[nSelAct] : CUDA_VALUE(1.0);
			cudafloat outn = output * M;
			
			cudafloat error = (desiredOutputs[n] - outn);
			
			if (m != nullptr) localGradientSpaceNet[nSelAct] = error * output * CUDA_SIGMOID_DERIVATE(M);
			
			outputs[n] = outn;
			
			localGradient[n] = error * M * CUDA_SIGMOID_DERIVATE(output);
			
			rms[PATTERN * NUM_NEURONS + NEURON] = error * error;
		}
	}
}

void KernelFireOutputLayer(cudaStream_t stream, dim3 & gridSize, int blockSize, cudafloat * inputs, cudafloat * weights, cudafloat * m, int mOffset, int totalNeuronsWithSelectiveActivation, cudafloat * desiredOutputs, cudafloat * outputs, cudafloat * localGradient, cudafloat * rms, cudafloat * localGradientSpaceNet, int numInputs) {
	switch(blockSize) {
		#ifdef FERMI
		case 1024:
			FireOutputLayerNeurons<1024><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, desiredOutputs, outputs, localGradient, rms, localGradientSpaceNet, numInputs);
			break;
		#endif
		case 512:
			FireOutputLayerNeurons<512><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, desiredOutputs, outputs, localGradient, rms, localGradientSpaceNet, numInputs);
			break;
		case 256:
			FireOutputLayerNeurons<256><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, desiredOutputs, outputs, localGradient, rms, localGradientSpaceNet, numInputs);
			break;
		case 128:
			FireOutputLayerNeurons<128><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, desiredOutputs, outputs, localGradient, rms, localGradientSpaceNet, numInputs);
			break;
		case 64:
			FireOutputLayerNeurons<64><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, desiredOutputs, outputs, localGradient, rms, localGradientSpaceNet, numInputs);
			break;
		case 32:
			FireOutputLayerNeurons<32><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, desiredOutputs, outputs, localGradient, rms, localGradientSpaceNet, numInputs);
			break;
		case 16:
			FireOutputLayerNeurons<16><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, desiredOutputs, outputs, localGradient, rms, localGradientSpaceNet, numInputs);
			break;
		case 8:
			FireOutputLayerNeurons<8><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, desiredOutputs, outputs, localGradient, rms, localGradientSpaceNet, numInputs);
			break;
		case 4:
			FireOutputLayerNeurons<4><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, desiredOutputs, outputs, localGradient, rms, localGradientSpaceNet, numInputs);
			break;
		case 2:
			FireOutputLayerNeurons<2><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, desiredOutputs, outputs, localGradient, rms, localGradientSpaceNet, numInputs);
			break;
		case 1:
			FireOutputLayerNeurons<1><<<gridSize, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, desiredOutputs, outputs, localGradient, rms, localGradientSpaceNet, numInputs);
			break;
	}
}

}