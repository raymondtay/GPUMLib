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

#include "RBMconfig.h"

namespace GPUMLib {

KERNEL InitBiasDeltasRBM(cudafloat * bias, cudafloat initialBias, cudafloat * lastDeltaW, cudafloat * lastDeltaB, cudafloat * lastDeltaWithoutLearningMomentumW, cudafloat * lastDeltaWithoutLearningMomentumB, cudafloat * learningRateW, cudafloat * learningRateB, cudafloat initialLearningRate, int weights, int J) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < weights) {
		lastDeltaW[idx] = CUDA_VALUE(0.0);

		learningRateW[idx] = initialLearningRate;
		lastDeltaWithoutLearningMomentumW[idx] = CUDA_VALUE(0.0);				

		if (idx < J) {
			bias[idx] = initialBias;
			lastDeltaB[idx] = CUDA_VALUE(0.0);
			lastDeltaWithoutLearningMomentumB[idx] = CUDA_VALUE(0.0);
			learningRateB[idx] = initialLearningRate;
		}
	}
}

KERNEL InitInputBiasDeltasRBM(cudafloat * v, cudafloat * bias, cudafloat * lastDeltaA, cudafloat * lastDeltaWithoutLearningMomentumA, cudafloat * learningRateA, cudafloat initialLearningRate, int I, int samples) {
	int input = blockIdx.x * blockDim.x + threadIdx.x;

	cudafloat sum = CUDA_VALUE(0.0);

	if (input < I) {
		for(int s = 0; s < samples; s++) sum += v[s * I + input];

		cudafloat pi = sum / samples;
		pi = Log(pi / (CUDA_VALUE(1.0) - pi));
		bias[input] = pi;

		lastDeltaA[input] = CUDA_VALUE(0.0);

		lastDeltaWithoutLearningMomentumA[input] = CUDA_VALUE(0.0);
		learningRateA[input] = initialLearningRate;
	}
}

}