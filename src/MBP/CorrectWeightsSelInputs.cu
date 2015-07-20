/*
	Noel Lopes is an Assistant Professor at the Polytechnic Institute of Guarda, Portugal (for more information see readme.txt)
	Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012 Noel de Jesus Mendonça Lopes

	This file is part of Multiple Back-Propagation.

	Multiple Back-Propagation is free software: you can redistribute it and/or modify
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

#define NEURON blockIdx.x
#define NUM_NEURONS gridDim.x

namespace GPUMLib {

template <int blockSize> KERNEL CorrectWeightsSelectiveInputs(cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, cudafloat * inputs, cudafloat * localGradient, cudafloat * selectiveNeuronsWeights, cudafloat * selectiveNeuronsBias, cudafloat * learningRateWeights, cudafloat * learningRateBias, cudafloat * lastDeltaWithoutLearningMomentumWeights, cudafloat * lastDeltaWithoutLearningMomentumBias, cudafloat * lastDeltaWeights, cudafloat * lastDeltaBias, cudafloat u, cudafloat d, cudafloat r, cudafloat maxStepSize, cudafloat momentum, int numberPatterns) {
	extern __shared__ cudafloat deltasWeights[];
	cudafloat * deltasBias = (deltasWeights + blockDim.x);
	
	if (bestRMS != NULL) {
		__shared__ cudafloat rms;
		__shared__ cudafloat bRMS;
	
		rms = *rmsF;
		bRMS = *bestRMS;
		if (rms >= bRMS * maxErrorGrowth) return;
	}

	deltasBias[threadIdx.x] = CUDA_VALUE(0.0);
	deltasWeights[threadIdx.x] = CUDA_VALUE(0.0);
	for(int p = threadIdx.x; p < numberPatterns; p += blockDim.x) {
		int n = p * NUM_NEURONS + NEURON;

		cudafloat i = inputs[n];
		if (!IsInfOrNaN(i)) {
			cudafloat delta = localGradient[n];

			deltasBias[threadIdx.x] += delta;
			deltasWeights[threadIdx.x] += delta * i;
		}
	}
	__syncthreads();

	if (blockSize >= 1024) {
		if (threadIdx.x < 512) {
			deltasBias[threadIdx.x] += deltasBias[threadIdx.x + 512];
			deltasWeights[threadIdx.x] += deltasWeights[threadIdx.x + 512];
		}
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (threadIdx.x < 256) {
			deltasBias[threadIdx.x] += deltasBias[threadIdx.x + 256];
			deltasWeights[threadIdx.x] += deltasWeights[threadIdx.x + 256];
		}
		__syncthreads();
	}	

	if (blockSize >= 256) {
		if (threadIdx.x < 128) {
			deltasBias[threadIdx.x] += deltasBias[threadIdx.x + 128];
			deltasWeights[threadIdx.x] += deltasWeights[threadIdx.x + 128];
		}
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadIdx.x < 64) {
			deltasBias[threadIdx.x] += deltasBias[threadIdx.x + 64];
			deltasWeights[threadIdx.x] += deltasWeights[threadIdx.x + 64];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		volatile cudafloat * _deltasBias = deltasBias;
		volatile cudafloat * _deltasWeights = deltasWeights;

		if (blockSize >= 64) {
			_deltasBias[threadIdx.x] += _deltasBias[threadIdx.x + 32];
			_deltasWeights[threadIdx.x] += _deltasWeights[threadIdx.x + 32];
		}		
	
		if (blockSize >= 32) {
			_deltasBias[threadIdx.x] += _deltasBias[threadIdx.x + 16];
			_deltasWeights[threadIdx.x] += _deltasWeights[threadIdx.x + 16];
		}
	
		if (blockSize >= 16) {
			_deltasBias[threadIdx.x] += _deltasBias[threadIdx.x + 8];
			_deltasWeights[threadIdx.x] += _deltasWeights[threadIdx.x + 8];
		}
	
		if (blockSize >= 8) {
			_deltasBias[threadIdx.x] += _deltasBias[threadIdx.x + 4];
			_deltasWeights[threadIdx.x] += _deltasWeights[threadIdx.x + 4];
		}
	
		if (blockSize >= 4) {
			_deltasBias[threadIdx.x] += _deltasBias[threadIdx.x + 2];
			_deltasWeights[threadIdx.x] += _deltasWeights[threadIdx.x + 2];
		}
		
		if (blockSize >= 2) {
			_deltasBias[threadIdx.x] += _deltasBias[threadIdx.x + 1];
			_deltasWeights[threadIdx.x] += _deltasWeights[threadIdx.x + 1];
		}
		
		if (threadIdx.x == 0) {
			cudafloat deltaB = deltasBias[0] / numberPatterns;
			cudafloat deltaW = deltasWeights[0] / numberPatterns;

			cudafloat learnRateB = learningRateBias[NEURON];
			cudafloat learnRateW = learningRateWeights[NEURON];

			cudafloat factorB = SAME_DIRECTION(lastDeltaWithoutLearningMomentumBias[NEURON], deltaB) ? u : d;
			cudafloat factorW = SAME_DIRECTION(lastDeltaWithoutLearningMomentumWeights[NEURON], deltaW) ? u : d;

			learnRateB *= factorB;
			learnRateW *= factorW;

			if (learnRateB > maxStepSize) learnRateB = maxStepSize;
			if (learnRateW > maxStepSize) learnRateW = maxStepSize;

			learningRateBias[NEURON] = learnRateB;
			learningRateWeights[NEURON] = learnRateW;

			lastDeltaWithoutLearningMomentumBias[NEURON] = deltaB;
			lastDeltaWithoutLearningMomentumWeights[NEURON] = deltaW;

			deltaB += momentum * lastDeltaBias[NEURON];
			deltaW += momentum * lastDeltaWeights[NEURON];

			lastDeltaBias[NEURON] = deltaB;
			lastDeltaWeights[NEURON] = deltaW;
			
			cudafloat wb = selectiveNeuronsBias[NEURON] + (learnRateB * deltaB);
			cudafloat w = selectiveNeuronsWeights[NEURON] + (learnRateW * deltaW);
			
			if (IsInfOrNaN(wb)) {
				lastDeltaBias[NEURON] = CUDA_VALUE(0.0);
				lastDeltaWithoutLearningMomentumBias[NEURON] = CUDA_VALUE(0.0);
				if (bestRMS != NULL) {
					learnRateB *= r;
					learningRateBias[NEURON] = learnRateB;
				}
			} else {
				selectiveNeuronsBias[NEURON] = wb;
			}

			if (IsInfOrNaN(w)) {
				lastDeltaWeights[NEURON] = CUDA_VALUE(0.0);
				lastDeltaWithoutLearningMomentumWeights[NEURON] = CUDA_VALUE(0.0);
				if (bestRMS != NULL) {
					learnRateW *= r;
					learningRateWeights[NEURON] = learnRateW;
				}
			} else {
				selectiveNeuronsWeights[NEURON] = w;
			}
		}
	}
}

#define CORRECT_WEIGHTS(X) CorrectWeightsSelectiveInputs<X><<<neurons, X, 2 * patterns * sizeof(cudafloat), stream>>>(rmsF, bestRMS, maxErrorGrowth, inputs, localGradient, selectiveNeuronsWeights, selectiveNeuronsBias, learningRateWeights, learningRateBias, lastDeltaWithoutLearningMomentumWeights, lastDeltaWithoutLearningMomentumBias, lastDeltaWeights, lastDeltaBias, u, d, r, maxStepSize, momentum, numberPatterns);

void KernelCorrectWeightsSelectiveInputs(cudaStream_t stream, int neurons, int patterns, cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, cudafloat * inputs, cudafloat * localGradient, cudafloat * selectiveNeuronsWeights, cudafloat * selectiveNeuronsBias, cudafloat * learningRateWeights, cudafloat * learningRateBias, cudafloat * lastDeltaWithoutLearningMomentumWeights, cudafloat * lastDeltaWithoutLearningMomentumBias, cudafloat * lastDeltaWeights, cudafloat * lastDeltaBias, cudafloat u, cudafloat d, cudafloat r, cudafloat maxStepSize, cudafloat momentum, int numberPatterns) {
	switch(patterns) {
		case 512:
			CORRECT_WEIGHTS(512);
			break;
		case 256:
			CORRECT_WEIGHTS(256);
			break;
		case 128:
			CORRECT_WEIGHTS(128);
			break;
		case 64:
			CORRECT_WEIGHTS(64);
			break;
		case 32:
			CORRECT_WEIGHTS(32);
			break;
		case 16:
			CORRECT_WEIGHTS(16);
			break;
		case 8:
			CORRECT_WEIGHTS(8);
			break;
		case 4:
			CORRECT_WEIGHTS(4);
			break;
		case 2:
			CORRECT_WEIGHTS(2);
			break;
		case 1:
			CORRECT_WEIGHTS(1);
			break;
	}
}

}