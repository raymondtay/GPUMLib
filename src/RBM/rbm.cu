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

#include <stdlib.h>

#include "../common/CudaDefinitions.h"
#include "../common/Utilities.h"
#include "../random/random.h"
#include "RBM.h"

namespace GPUMLib {

KERNEL InitBiasDeltasRBM(cudafloat * bias, cudafloat initialBias, cudafloat * lastDeltaW, cudafloat * lastDeltaB, cudafloat * lastDeltaWithoutLearningMomentumW, cudafloat * lastDeltaWithoutLearningMomentumB, cudafloat * learningRateW, cudafloat * learningRateB, cudafloat initialLearningRate, int weights, int J);
KERNEL InitInputBiasDeltasRBM(cudafloat * v, cudafloat * bias, cudafloat * lastDeltaA, cudafloat * lastDeltaWithoutLearningMomentumA, cudafloat * learningRateA, cudafloat initialLearningRate, int I, int samples);
KERNEL CorrectWeightsRBM(cudafloat * v_data, cudafloat * h_data, cudafloat * v_recon, cudafloat * h_recon, int samples, cudafloat * learningRateW, cudafloat * lastDeltaWithoutLearningMomentumW, cudafloat * lastDeltaW, cudafloat * learningRateB, cudafloat * lastDeltaWithoutLearningMomentumB, cudafloat * lastDeltaB, cudafloat * learningRateA, cudafloat * lastDeltaWithoutLearningMomentumA, cudafloat * lastDeltaA, cudafloat u, cudafloat d, cudafloat momentum, cudafloat * weights, cudafloat * b, cudafloat * a, cudafloat * errors, int I, int J);
	
void KernelComputeStatusVisibleUnitsRBM(dim3 & gridSize, int blockSize, cudafloat * h, cudafloat * weights, cudafloat * a, cudafloat * v, float * randomValues, int J);
void KernelComputeStatusHiddenUnitsRBM(dim3 & gridSize, int blockSize, cudafloat * v, cudafloat * weights, cudafloat * b, cudafloat * h, float * randomValues, int I);
KERNEL ComputeStatusHiddenUnitsSmallRBM(cudafloat * v, cudafloat * weights, cudafloat * b, cudafloat * h, float * randomValues);
KERNEL ComputeStatusVisibleUnitsSmallRBM(cudafloat * h, cudafloat * weights, cudafloat * a, cudafloat * v, float * randomValues);

void RBM::RandomizeWeights() {
	int nWeights = w.Elements();

	cudafloat * weights = w.HostPointer();
			
	for (int i = 0; i < nWeights; i++) weights[i] = CUDA_VALUE(2.0) * stdWeights * ((cudafloat) rand() / RAND_MAX) - stdWeights;
	w.UpdateDevice();

	int blockSize = NumberThreadsPerBlockThatBestFit(nWeights);
	int blocks = NumberBlocks(nWeights, blockSize);

	InitBiasDeltasRBM<<<blocks, blockSize>>>(b.DevicePointer(), INITIAL_BIAS_HIDDEN_UNITS, lastDelta.w.Pointer(), lastDelta.b.Pointer(), lastDeltaWithoutLearningMomentum.w.Pointer(), lastDeltaWithoutLearningMomentum.b.Pointer(), learningRate.w.Pointer(), learningRate.b.Pointer(), initialLearningRate, nWeights, J);

	blocks = NumberBlocks(I, inputsBlockSize);

	InitInputBiasDeltasRBM<<<blocks, inputsBlockSize>>>(v.Pointer(),  a.DevicePointer(), lastDelta.a.Pointer(), lastDeltaWithoutLearningMomentum.a.Pointer(), learningRate.a.Pointer(), initialLearningRate, I, samples);

	epoch = 0;
}

void RBM::ComputeStatusUnits(cudafloat * v, cudafloat * h, cudafloat * v_reconstructed, int samples, float * rnd) {
	int connections = w.Elements();

	dim3 dimJsamples;
	dimJsamples.x = J;
	dimJsamples.y = samples;

	if(connections > MAX_THREADS_PER_BLOCK) {
		KernelComputeStatusHiddenUnitsRBM(dimJsamples, inputsBlockSize, v, w.DevicePointer(), b.DevicePointer(), h, rnd, I);
	} else {
		ComputeStatusHiddenUnitsSmallRBM<<<samples, dimIJ, connections * sizeof(cudafloat)>>>(v, w.DevicePointer(), b.DevicePointer(), h, rnd);
	}

	dim3 dimIsamples;
	dimIsamples.x = I;
	dimIsamples.y = samples;

	if (v_reconstructed != nullptr) {
		rnd = (useBinaryValuesVisibleReconstruction) ? (rnd + J * samples) : nullptr;

		if(connections > MAX_THREADS_PER_BLOCK) {
			KernelComputeStatusVisibleUnitsRBM(dimIsamples, hiddenUnitsBlockSize, h, w.DevicePointer(), a.DevicePointer(), v_reconstructed, rnd, J);
		} else {		
			ComputeStatusVisibleUnitsSmallRBM<<<samples, dimJI, connections * sizeof(cudafloat)>>>(h, w.DevicePointer(), a.DevicePointer(), v_reconstructed, rnd);
		}
	}
}

void RBM::ContrastiveDivergence(int n) {
	int sizeLastBatch = samples;
	int batches = 1;

	if (miniBatchSize > 0) {
		batches = samples / miniBatchSize;
		sizeLastBatch = samples % miniBatchSize;
		if (sizeLastBatch > 0) {
			batches++; 
		} else {
			sizeLastBatch = miniBatchSize;
		}
	}

	dim3 block;
	block.x = 16;
	block.y = 16;
	
	dim3 grid;
	grid.x = NumberBlocks(I, block.x);
	grid.y = NumberBlocks(J, block.y);

	cudafloat * vd = v.Pointer();
	cudafloat * hd = h_data.Pointer();
	cudafloat * vr = v_recon.Pointer();
	cudafloat * hr = h_recon.Pointer();
	cudafloat * cerrors = errors.Pointer();

	Random::Fill(randomValues);
	float * rnd = randomValues.Pointer();

	int lastBatch = batches - 1;
	for(int batch = 0; batch < batches; batch++) {
		int samples = (batch == lastBatch) ? sizeLastBatch : miniBatchSize;

		ComputeStatusUnits(vd, hd, vr, samples, rnd);
		rnd += samples * (useBinaryValuesVisibleReconstruction ? (I + J) : J);

		for (int k = 1; k < n; k++) {
			ComputeStatusUnits(vr, hr, vr, samples, rnd);
			rnd += samples * (useBinaryValuesVisibleReconstruction ? (I + J) : J);
		}

		ComputeStatusUnits(vr, hr, nullptr, samples, nullptr);

		CorrectWeightsRBM<<<grid, block>>>(vd, hd, vr, hr, samples, learningRate.w.Pointer(), lastDeltaWithoutLearningMomentum.w.Pointer(), lastDelta.w.Pointer(), learningRate.b.Pointer(), lastDeltaWithoutLearningMomentum.b.Pointer(), lastDelta.b.Pointer(), learningRate.a.Pointer(), lastDeltaWithoutLearningMomentum.a.Pointer(), lastDelta.a.Pointer(), U_FACTOR, D_FACTOR, momentum, w.DevicePointer(), b.DevicePointer(), a.DevicePointer(), cerrors, I, J);

		vd += miniBatchSize;
		hd += miniBatchSize;
		vr += miniBatchSize;
		hr += miniBatchSize;
		cerrors += miniBatchSize;
	}

	epoch++;
}

}