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

#include "MBPkernels.h"

#define OUTPUT_NEURON threadIdx.x
#define OUTPUT_INCLUDING_BIAS (threadIdx.x + 1)
#define NUM_OUTPUTS blockDim.x

#define NEURON threadIdx.y
#define NUM_NEURONS blockDim.y

#define NUM_INPUTS_OUTPUT_NEURON (NUM_NEURONS + 1)

#define PATTERN blockIdx.x

namespace GPUMLib {

KERNEL CalculateLocalGradient(cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, cudafloat * outputs, cudafloat * weights, cudafloat * m, int mOffset, int totalNeuronsWithSelectiveActivation, cudafloat * localGradientNextLayer, cudafloat * localGradient, cudafloat * localGradientSpaceNet) {
	extern __shared__ cudafloat lg[];
	
	if (bestRMS != nullptr) {
		__shared__ cudafloat rms;
		__shared__ cudafloat bRMS;
		
		rms = *rmsF;
		bRMS = *bestRMS;
		if (rms >= bRMS * maxErrorGrowth) return;
	}

	cudafloat * lgNextLayer = (lg + (NUM_OUTPUTS * NUM_NEURONS));
			
	if (NEURON == 0) lgNextLayer[OUTPUT_NEURON] = localGradientNextLayer[PATTERN * NUM_OUTPUTS + OUTPUT_NEURON];
	
	int connection = OUTPUT_NEURON * NUM_INPUTS_OUTPUT_NEURON + NEURON + 1;    
	int threadId = (NEURON * NUM_OUTPUTS + OUTPUT_NEURON);
	
	__syncthreads();    
	
	lg[threadId] = weights[connection] * lgNextLayer[OUTPUT_NEURON];
	__syncthreads();

	int numberElemSum = NUM_OUTPUTS;
	for(int sumUpTo = (numberElemSum >> 1); numberElemSum > 1; sumUpTo = (numberElemSum >> 1)) {
		int nextNumberElemSum = sumUpTo;
		if (numberElemSum & 1) nextNumberElemSum++;
	
		if (OUTPUT_NEURON < sumUpTo) lg[threadId] += lg[threadId + nextNumberElemSum];
		
		numberElemSum = nextNumberElemSum;
		
		__syncthreads();
	}
	
	if (OUTPUT_NEURON == 0) {
		int n = PATTERN * NUM_NEURONS + NEURON;
		
		cudafloat Fh = outputs[n];
	
		cudafloat lgn = lg[threadId];
	
		if (m != nullptr) {
			int nSelAct = PATTERN * totalNeuronsWithSelectiveActivation + NEURON + mOffset;

			cudafloat M = m[nSelAct];
			if (M == CUDA_VALUE(0.0)) {
				localGradientSpaceNet[nSelAct] = CUDA_VALUE(0.0);
			} else {
				Fh = Fh / M;
				localGradientSpaceNet[nSelAct] = lgn * Fh * CUDA_SIGMOID_DERIVATE(M);
			}
			lgn *= M;
		}
	
		localGradient[n] = lgn * CUDA_SIGMOID_DERIVATE(Fh);
	}
}

}