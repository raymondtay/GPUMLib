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
#include "../reduction/SumWarp.h"

namespace GPUMLib {

__device__ __forceinline__ void UpdateWeight(cudafloat learningRate, cudafloat momentum, cudafloat delta, cudafloat * lastDelta, cudafloat * lastDeltaWithoutLearningMomentum, cudafloat * weights, int w) {
	momentum *= learningRate;
	if (momentum < CUDA_VALUE(0.1)) momentum = CUDA_VALUE(0.1);
	if (momentum > CUDA_VALUE(0.9)) momentum = CUDA_VALUE(0.9);

	cudafloat neww = weights[w] + learningRate * delta + momentum * lastDelta[w];
	delta += momentum * lastDelta[w];

	if (IsInfOrNaN(neww)) {
		delta = CUDA_VALUE(0.0);
		lastDeltaWithoutLearningMomentum[w] = CUDA_VALUE(0.0);
	} else {				
		weights[w] = neww;
	}

	lastDelta[w] = delta;
}

__device__ __forceinline__ cudafloat UpdateLearningRate(cudafloat * lr, cudafloat * lastDeltaWithoutLearningMomentum, cudafloat delta, int w, cudafloat u, cudafloat d) {
	cudafloat learningRate = lr[w];
	
	learningRate *= (SAME_DIRECTION(lastDeltaWithoutLearningMomentum[w], delta) ? u : d);	
	if (learningRate > MAX_STEP_SIZE) learningRate = MAX_STEP_SIZE;

	lr[w] = learningRate;
	lastDeltaWithoutLearningMomentum[w] = delta;

	return learningRate;
}

KERNEL CorrectWeightsRBM(cudafloat * v_data, cudafloat * h_data, cudafloat * v_recon, cudafloat * h_recon, int samples, cudafloat * learningRateW, cudafloat * lastDeltaWithoutLearningMomentumW, cudafloat * lastDeltaW, cudafloat * learningRateB, cudafloat * lastDeltaWithoutLearningMomentumB, cudafloat * lastDeltaB, cudafloat * learningRateA, cudafloat * lastDeltaWithoutLearningMomentumA, cudafloat * lastDeltaA, cudafloat u, cudafloat d, cudafloat momentum, cudafloat * weights, cudafloat * b, cudafloat * a, cudafloat * errors, int I, int J) {
	__shared__ cudafloat vd[16];
	__shared__ cudafloat vr[16];
	__shared__ cudafloat hd[16];
	__shared__ cudafloat hr[16];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	cudafloat error  = CUDA_VALUE(0.0);
	cudafloat deltaW = CUDA_VALUE(0.0);
	cudafloat deltaB = CUDA_VALUE(0.0);
	cudafloat deltaA = CUDA_VALUE(0.0);

	for (int s = 0; s < samples; s++) {		
		if (threadIdx.y == 0 && i < I) {
			cudafloat dat = v_data[s * I + i];
			cudafloat rec = v_recon[s * I + i];

			vd[threadIdx.x] = dat;
			vr[threadIdx.x] = rec;

			cudafloat e = dat - rec;
			deltaA += e;

			error += e * e;
		}

		if (threadIdx.x == 0 && j < J) {
			cudafloat dat = h_data[s * J + j];
			cudafloat rec = h_recon[s * J + j];

			hd[threadIdx.y] = dat;
			hr[threadIdx.y] = rec;

			deltaB += dat - rec;
		}

		__syncthreads();

		deltaW += vd[threadIdx.x] * hd[threadIdx.y] - vr[threadIdx.x] * hr[threadIdx.y];
	}

	// update weights
	if (i < I && j < J) {
		deltaW /= samples;

		int w = j * I + i;

		cudafloat learningRate = UpdateLearningRate(learningRateW, lastDeltaWithoutLearningMomentumW, deltaW, w, u, d);
		UpdateWeight(learningRate, momentum, deltaW, lastDeltaW, lastDeltaWithoutLearningMomentumW, weights, w);
	}

	if(i < I && threadIdx.y == 0) {
		errors[i] = error;

		// Update a
		if (j == 0) {
			deltaA /= samples;

			cudafloat learningRate = UpdateLearningRate(learningRateA, lastDeltaWithoutLearningMomentumA, deltaA, i, u, d);
			UpdateWeight(learningRate, momentum, deltaA, lastDeltaA, lastDeltaWithoutLearningMomentumA, a, i);
		}
	}

	// Update b
	if (i == 0 && j < J) {
		deltaB /= samples;

		cudafloat learningRate = UpdateLearningRate(learningRateB, lastDeltaWithoutLearningMomentumB, deltaB, j, u, d);
		UpdateWeight(learningRate, momentum, deltaB, lastDeltaB, lastDeltaWithoutLearningMomentumB, b, j);
	}
}

}