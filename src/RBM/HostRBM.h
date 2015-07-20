/*
	Noel Lopes is an Assistant Professor at the Polytechnic Institute of Guarda, Portugal
	Copyright (C) 2009, 2010, 2011 Noel de Jesus Mendonça Lopes

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

#ifndef RBMH_h
#define RBMH_h

#include <stdlib.h>
#include <math.h>
#include <float.h> 
#include <limits>

#include "RBMconfig.h"

#include "../memory/HostArray.h"

namespace GPUMLib {

//! \addtogroup rbmh Restricted Boltzman Machine Host (CPU) class
//! @{

//! Represents a Restricted Boltzman Machine (Host - CPU).
class RBMhost {
	friend class DBNhost;

	private:
		int I; // number of visible units
		int J; // number of hidden units
		int samples;
		
		HostArray<cudafloat> a; // bias of the visible layer
		HostArray<cudafloat> b; // bias of the hidden layer		
		HostMatrix<cudafloat> w; // weights

		HostArray<cudafloat> lastDeltaA; 
		HostArray<cudafloat> lastDeltaB; 
		HostMatrix<cudafloat> lastDeltaW;
		
		HostMatrix<cudafloat> * v; // visible units / inputs

		HostArray<cudafloat> learningRateA;
		HostArray<cudafloat> learningRateB;
		HostMatrix<cudafloat> learningRateW;

		HostArray<cudafloat> lastDeltaWithoutLearningMomentumA; 
		HostArray<cudafloat> lastDeltaWithoutLearningMomentumB; 
		HostMatrix<cudafloat> lastDeltaWithoutLearningMomentumW;

		cudafloat stdWeights;
		cudafloat initialLearningRate;
		cudafloat momentum;

		bool useBinaryValuesVisibleReconstruction;
		
		HostMatrix<cudafloat> h_data;
		HostMatrix<cudafloat> h_recon;
		HostMatrix<cudafloat> v_recon;

		int epoch;

		// Used by DBNs
		RBMhost(RBMhost * previousLayer, int hiddenUnits, cudafloat initialLearningRate, cudafloat momentum, bool useBinaryValuesVisibleReconstruction, cudafloat stdWeights) {
			v = &previousLayer->h_data;

			this->stdWeights = stdWeights;
			this->initialLearningRate = initialLearningRate;			
			this->useBinaryValuesVisibleReconstruction = useBinaryValuesVisibleReconstruction;

			I = previousLayer->J;
			J = hiddenUnits;
					
			w.ResizeWithoutPreservingData(hiddenUnits, I);
			a.ResizeWithoutPreservingData(I);
			b.ResizeWithoutPreservingData(hiddenUnits);

			learningRateW.ResizeWithoutPreservingData(hiddenUnits, I);
			learningRateA.ResizeWithoutPreservingData(I);
			learningRateB.ResizeWithoutPreservingData(hiddenUnits);

			lastDeltaWithoutLearningMomentumW.ResizeWithoutPreservingData(hiddenUnits, I);
			lastDeltaWithoutLearningMomentumA.ResizeWithoutPreservingData(I);
			lastDeltaWithoutLearningMomentumB.ResizeWithoutPreservingData(hiddenUnits);

			lastDeltaW.ResizeWithoutPreservingData(hiddenUnits, I);
			lastDeltaA.ResizeWithoutPreservingData(I);
			lastDeltaB.ResizeWithoutPreservingData(hiddenUnits);

			samples = v->Rows();
			
			h_data.ResizeWithoutPreservingData(samples, hiddenUnits);
			h_recon.ResizeWithoutPreservingData(samples, hiddenUnits);
			v_recon.ResizeWithoutPreservingData(samples, I);

			this->momentum = momentum;

			epoch = 0;
		}

	public:
		//! Constructs a Restricted Boltzman Machine that can be trained using the CPU (Host).
		//! \param visibleUnits Number of inputs.
		//! \param hiddenUnits Number of hidden neurons.
		//! \param inputs Inputs of the training dataset. Each row of the matrix should contain a pattern (sample) and each column an input.
		//! \param initialLearningRate Initial learning rate.
		//! \param momentum Momentum (optional).
		//! \param useBinaryValuesVisibleReconstruction Use binary values for the visibible layer reconstruction (optional, default = false)
		//! \param stdWeights Defines the maximum and minimum value for the weights. The weights will be initialized with a random number between -stdWeights and stdWeights.
		RBMhost(int visibleUnits, int hiddenUnits, HostMatrix<cudafloat> & inputs, cudafloat initialLearningRate, cudafloat momentum = DEFAULT_MOMENTUM, bool useBinaryValuesVisibleReconstruction = false, cudafloat stdWeights = STD_WEIGHTS) {
			assert(visibleUnits == inputs.Columns());

			v = &inputs;

			this->useBinaryValuesVisibleReconstruction = useBinaryValuesVisibleReconstruction;

			I = visibleUnits;
			J = hiddenUnits;
					
			w.ResizeWithoutPreservingData(hiddenUnits, visibleUnits);
			a.ResizeWithoutPreservingData(visibleUnits);
			b.ResizeWithoutPreservingData(hiddenUnits);

			learningRateW.ResizeWithoutPreservingData(hiddenUnits, visibleUnits);
			learningRateA.ResizeWithoutPreservingData(visibleUnits);
			learningRateB.ResizeWithoutPreservingData(hiddenUnits);

			lastDeltaWithoutLearningMomentumW.ResizeWithoutPreservingData(hiddenUnits, visibleUnits);
			lastDeltaWithoutLearningMomentumA.ResizeWithoutPreservingData(visibleUnits);
			lastDeltaWithoutLearningMomentumB.ResizeWithoutPreservingData(hiddenUnits);

			lastDeltaW.ResizeWithoutPreservingData(hiddenUnits, visibleUnits);
			lastDeltaA.ResizeWithoutPreservingData(visibleUnits);
			lastDeltaB.ResizeWithoutPreservingData(hiddenUnits);

			samples = inputs.Rows();
			
			h_data.ResizeWithoutPreservingData(samples, hiddenUnits);
			h_recon.ResizeWithoutPreservingData(samples, hiddenUnits);
			v_recon.ResizeWithoutPreservingData(samples, visibleUnits);

			RandomizeWeights(stdWeights, initialLearningRate);
			this->momentum = momentum;			
		}

		//! Randomizes the weights of the RBM, between -stdWeights and stdWeights.
		//! \param stdWeights Defines the maximum and minimum value for the weights.
		void RandomizeWeights(cudafloat stdWeights) {
			this->stdWeights = stdWeights;
			RandomizeWeights();
		}

		//! Randomizes the weights of the RBM, between -stdWeights and stdWeights.
		//! \param stdWeights Defines the maximum and minimum value for the weights.
		//! \param initialLearningRate Initial learning rate.
		void RandomizeWeights(cudafloat stdWeights, cudafloat initialLearningRate) {
			this->initialLearningRate = initialLearningRate;
			RandomizeWeights(stdWeights);
		}

		//! Randomizes the weights of the RBM
		void RandomizeWeights() {
			int nWeights = w.Elements();

			cudafloat * weights = w.Pointer();
			cudafloat * lastDelta = lastDeltaW.Pointer();
			
			cudafloat * lastDeltaWM = lastDeltaWithoutLearningMomentumW.Pointer();
			cudafloat * stepSizes = learningRateW.Pointer();

			for (int w = 0; w < nWeights; w++) {
				weights[w] = CUDA_VALUE(2.0) * stdWeights * ((cudafloat) rand() / RAND_MAX) - stdWeights;
				lastDelta[w] = CUDA_VALUE(0.0);				
				lastDeltaWM[w] = CUDA_VALUE(0.0);
				stepSizes[w] = initialLearningRate;
			}

			for (int j = 0; j < J; j++) {
				b[j] = INITIAL_BIAS_HIDDEN_UNITS;
				lastDeltaB[j] = CUDA_VALUE(0.0);				
				lastDeltaWithoutLearningMomentumB[j] = CUDA_VALUE(0.0);
				learningRateB[j] = initialLearningRate;
			}
			
			for (int i = 0; i < I; i++) {
				a[i] = CUDA_VALUE(0.0);
				lastDeltaA[i] = CUDA_VALUE(0.0);				
				lastDeltaWithoutLearningMomentumA[i] = CUDA_VALUE(0.0);
				learningRateA[i] = initialLearningRate;
			}

			for(int s = 0; s < samples; s++) {
				for (int i = 0; i < I; i++) {
					if ((*v)(s, i) > CUDA_VALUE(0.0)) a[i]++;
				}
			}

			for (int i = 0; i < I; i++) {
				cudafloat pi = a[i] / samples;
				a[i] = log(pi / (CUDA_VALUE(1.0) - pi));
				if (IsInfOrNaN(a[i])) a[i] = CUDA_VALUE(0.0);
			}

			epoch = 0;
		}
		
		void ContrastiveDivergence(int n) {
			ComputeStatusHiddenUnits(*v, h_data);
			ComputeStatusVisibleUnits(h_data, v_recon);

			for (int k = 1; k < n; k++) {
				ComputeStatusHiddenUnits(v_recon, h_recon);
				ComputeStatusVisibleUnits(h_recon, v_recon);
			}

			ComputeStatusHiddenUnits(v_recon, h_recon, false);

			for (int j = 0; j < J; j++) {
				for(int i = 0; i < I; i++) {
					cudafloat delta = CUDA_VALUE(0.0);

					for (int s = 0; s < v->Rows(); s++) {
						delta += (*v)(s, i) * h_data(s, j) - v_recon(s, i) * h_recon(s, j);
					}

					delta /= samples;

					cudafloat learnRate = learningRateW(j, i);
					cudafloat factor = SAME_DIRECTION(lastDeltaWithoutLearningMomentumW(j, i), delta) ? U_FACTOR : D_FACTOR;
					learnRate *= factor;
					if (learnRate > MAX_STEP_SIZE) learnRate = MAX_STEP_SIZE;
					learningRateW(j, i) = learnRate;

					lastDeltaWithoutLearningMomentumW(j, i) = delta;

					cudafloat momentum = this->momentum * learnRate;
					if (momentum < CUDA_VALUE(0.1)) momentum = CUDA_VALUE(0.1);
					if (momentum > CUDA_VALUE(0.9)) momentum = CUDA_VALUE(0.9);

					delta += momentum * lastDeltaW(j, i);
					lastDeltaW(j, i) = delta;

					cudafloat neww = w(j, i)  + (learnRate * delta);
					if (IsInfOrNaN(neww)) {
						lastDeltaW(j, i) = CUDA_VALUE(0.0);
						lastDeltaWithoutLearningMomentumW(j, i) = CUDA_VALUE(0.0);
					} else {
						w(j, i) = neww;
					}
				}

				cudafloat deltaB = CUDA_VALUE(0.0);
				for (int s = 0; s < v->Rows(); s++) {
					deltaB += h_data(s, j) - h_recon(s, j);
				}

				deltaB /= samples;

				cudafloat learnRate = learningRateB[j];
				cudafloat factor = SAME_DIRECTION(lastDeltaWithoutLearningMomentumB[j], deltaB) ? U_FACTOR : D_FACTOR;
				learnRate *= factor;
				if (learnRate > MAX_STEP_SIZE) learnRate = MAX_STEP_SIZE;
				learningRateB[j] = learnRate;

				lastDeltaWithoutLearningMomentumB[j] = deltaB;

				cudafloat momentum = this->momentum * learnRate;
				if (momentum < CUDA_VALUE(0.1)) momentum = CUDA_VALUE(0.1);
				if (momentum > CUDA_VALUE(0.9)) momentum = CUDA_VALUE(0.9);

				deltaB += momentum * lastDeltaB[j];
				lastDeltaB[j] = deltaB;

				cudafloat newb = b[j]  + (learnRate * deltaB);
				if (IsInfOrNaN(newb)) {
					lastDeltaB[j] = CUDA_VALUE(0.0);
					lastDeltaWithoutLearningMomentumB[j] = CUDA_VALUE(0.0);
				} else {
					b[j] = newb;
				}
			}

			for(int i = 0; i < I; i++) {
				cudafloat deltaA = CUDA_VALUE(0.0);

				for (int s = 0; s < v->Rows(); s++) {
					deltaA += (*v)(s, i) - v_recon(s, i);
				}

				deltaA /= samples;

				cudafloat learnRate = learningRateA[i];
				cudafloat factor = SAME_DIRECTION(lastDeltaWithoutLearningMomentumA[i], deltaA) ? U_FACTOR : D_FACTOR;
				learnRate *= factor;
				if (learnRate > MAX_STEP_SIZE) learnRate = MAX_STEP_SIZE;
				learningRateA[i] = learnRate;

				lastDeltaWithoutLearningMomentumA[i] = deltaA;

				cudafloat momentum = this->momentum * learnRate;
				if (momentum < CUDA_VALUE(0.1)) momentum = CUDA_VALUE(0.1);
				if (momentum > CUDA_VALUE(0.9)) momentum = CUDA_VALUE(0.9);

				deltaA += momentum * lastDeltaA[i];
				lastDeltaA[i] = deltaA;

				cudafloat newa = a[i]  + (learnRate * deltaA);
				if (IsInfOrNaN(newa)) {
					lastDeltaA[i] = CUDA_VALUE(0.0);
					lastDeltaWithoutLearningMomentumA[i] = CUDA_VALUE(0.0);
				} else {
					a[i] = newa;
				}
			}

			epoch++;
		}

		//! Gets the mean square error of the RBM.
		//! \return The mean square error.
		cudafloat MeanSquareError() const {
			assert(epoch > 0);

			cudafloat error;

			error = CUDA_VALUE(0.0);
			for (int s = 0; s < (*v).Rows(); s++) {
				for(int i = 0; i < I; i++) {
					cudafloat e = (*v)(s, i) - v_recon(s, i);
					error += e * e;
				}
			}

			return error / ((cudafloat) samples * I);
		}

		//! Gets the current training epoch.
		//! \return The current epoch.
		int Epoch() const {
			return epoch;
		}

	private:
		static cudafloat Binarize(cudafloat probability) {
			return (probability > ((cudafloat) rand() / RAND_MAX)) ? CUDA_VALUE(1.0) : CUDA_VALUE(0.0);
		}

		void ComputeStatusHiddenUnits(HostMatrix<cudafloat> & v, HostMatrix<cudafloat> & h, bool binary = true) {
			for (int s = 0; s < samples; s++) {
				for (int j = 0; j < J; j++) {
					cudafloat sum = b[j];
					for(int i = 0; i < I; i++) sum += v(s, i) * w(j, i);

					cudafloat result = CUDA_SIGMOID(sum);
					if (binary) result = Binarize(result);
					h(s, j) = result;
				}
			}
		}

		void ComputeStatusVisibleUnits(HostMatrix<cudafloat> & h, HostMatrix<cudafloat> & v) {
			for (int s = 0; s < samples; s++) {
				for(int i = 0; i < I; i++) {
					cudafloat sum = a[i];
					for (int j = 0; j < J; j++) sum += h(s, j) * w(j, i);

					cudafloat result = CUDA_SIGMOID(sum);
					if (useBinaryValuesVisibleReconstruction) result = Binarize(result);
					v(s, i) = result;
				}
			}
		}

	public:
		//! Gets the weights matrix
		//! \return the weights matrix
		HostMatrix<cudafloat> GetWeights() {
			return HostMatrix<cudafloat>(w);
		}

		//! Gets the visible units bias
		//! \return an array with the bias
		HostArray<cudafloat> GetVisibleBias() {
			return HostArray<cudafloat>(a);
		}
		
		//! Gets the hidden units bias
		//! \return an array with the bias
		HostArray<cudafloat> GetHiddenBias() {
			return HostArray<cudafloat>(b);
		}

		HostMatrix<cudafloat> GetOutputs() {
			ComputeStatusHiddenUnits(*v, h_data);
			return HostMatrix<cudafloat>(h_data);
		}
};

//! \example DBNapp.cpp 
//! Example of the DBN and RBM algorithms usage.

//! @}

}

#endif