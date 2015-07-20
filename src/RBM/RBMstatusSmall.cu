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
#include "../common/CudaDefinitions.h"

namespace GPUMLib {

#define INPUT threadIdx.x
#define NUM_INPUTS blockDim.x

#define NEURON threadIdx.y
#define NUM_NEURONS blockDim.y

#define SAMPLE blockIdx.x

KERNEL ComputeStatusHiddenUnitsSmallRBM(cudafloat * v, cudafloat * weights, cudafloat * b, cudafloat * h, float * randomValues) {
	extern __shared__ cudafloat iw[];

	int connection = NEURON * NUM_INPUTS + INPUT;

	/*******
	For each each input connection of all layer neurons, calculate the weight * input.
	Results will be held in iw[]. This is done for the current sample.
	*******/
	cudafloat w = weights[connection];
	iw[connection] = w * v[SAMPLE * NUM_INPUTS + INPUT];
	__syncthreads();

	/*******
	For each layer neuron, calculate its activation: sum(weight * input).
	Results for neuron n will held on iw[n * NUM_INPUTS].
	This is done for the current sample.
	*******/
	int numberElemSum = NUM_INPUTS;
	for(int sumUpTo = (numberElemSum >> 1); numberElemSum > 1; sumUpTo = (numberElemSum >> 1)) {
		int nextNumberElemSum = sumUpTo;
		if (numberElemSum & 1) nextNumberElemSum++;
	
		if (INPUT < sumUpTo) iw[connection] += iw[connection + nextNumberElemSum];
		numberElemSum = nextNumberElemSum;
		
		__syncthreads();
	}

	/*******
	Calculate the neurons output
	*******/
	__shared__ cudafloat output;
	if (INPUT == 0) {
		output = CUDA_SIGMOID(iw[connection] + b[NEURON]);
		int idx = SAMPLE * NUM_NEURONS + NEURON;
		if (randomValues != nullptr) output = (output > randomValues[idx]) ? CUDA_VALUE(1.0) : CUDA_VALUE(0.0);		
		h[idx] = output;
	}
}

KERNEL ComputeStatusVisibleUnitsSmallRBM(cudafloat * h, cudafloat * weights, cudafloat * a, cudafloat * v, float * randomValues) {
	extern __shared__ cudafloat sum[];

	int connection = NEURON * NUM_INPUTS + INPUT;

	sum[connection] = h[SAMPLE * NUM_INPUTS + INPUT] * weights[INPUT * NUM_NEURONS + NEURON];
	__syncthreads();

	///*******
	//For each layer neuron, calculate its activation
	//Results for neuron n will held on sum[n * NUM_INPUTS].
	//This is done for the current sample.
	//*******/
	int numberElemSum = NUM_INPUTS;
	for(int sumUpTo = (numberElemSum >> 1); numberElemSum > 1; sumUpTo = (numberElemSum >> 1)) {
		int nextNumberElemSum = sumUpTo;
		if (numberElemSum & 1) nextNumberElemSum++;
	
		if (INPUT < sumUpTo) sum[connection] += sum[connection + nextNumberElemSum];
		numberElemSum = nextNumberElemSum;
		
		__syncthreads();
	}

	///*******
	//Calculate the neurons output
	//*******/
	if (INPUT == 0) {
		cudafloat output = CUDA_SIGMOID(sum[connection] + a[NEURON]);
		int idx = SAMPLE * NUM_NEURONS + NEURON;
		if (randomValues != nullptr) output = (output > randomValues[idx]) ? CUDA_VALUE(1.0) : CUDA_VALUE(0.0);
		v[idx] = output;
	}
}

}
