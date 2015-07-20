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

#ifndef GPUMLib_RBM_h
#define GPUMLib_RBM_h

#include "RBMconfig.h"

#include "../memory/DeviceAccessibleVariable.h"
#include "../common/CudaStreams.h"
#include "../common/Utilities.h"
#include "../memory/CudaArray.h"
#include "../memory/CudaMatrix.h"
#include "../reduction/reduction.h"

#include <iostream>
#include <ostream>

using namespace std;

namespace GPUMLib {

//! \addtogroup rbm Restricted Boltzman Machine (GPU) class
//! @{

//! Represents a Restricted Boltzman Machine (GPU).
class RBM {
	friend class DBN;

	private:
		class ConnectionsInfo {
			public:
				DeviceArray<cudafloat> a; // bias of the visible layer
				DeviceArray<cudafloat> b; // bias of the hidden layer
				DeviceMatrix<cudafloat> w; // weights

				bool Resize(int I, int J) {
					if (w.ResizeWithoutPreservingData(J, I) != J * I) return false;
					if (a.ResizeWithoutPreservingData(I) != I) return false;
					if (b.ResizeWithoutPreservingData(J) != J) return false;

					return true;
				}

				void Dispose() {
					a.Dispose();
					b.Dispose();
					w.Dispose();
				}
		};

		int I; // number of visible units
		int J; // number of hidden units
		int samples;
		int miniBatchSize;

		DeviceMatrix<cudafloat> v; // visible units inputs
		
		CudaArray<cudafloat> a; // bias of the visible layer
		CudaArray<cudafloat> b; // bias of the hidden layer
		CudaMatrix<cudafloat> w; // weights

		ConnectionsInfo lastDelta;
		ConnectionsInfo learningRate;
		ConnectionsInfo lastDeltaWithoutLearningMomentum;

		cudafloat stdWeights;
		cudafloat initialLearningRate;
		cudafloat momentum;

		bool useBinaryValuesVisibleReconstruction;
		
		DeviceMatrix<cudafloat> h_data;
		DeviceMatrix<cudafloat> h_recon;
		DeviceMatrix<cudafloat> v_recon;

		DeviceArray<float> randomValues;

		int epoch;

		dim3 dimIJ;
		dim3 dimJI;

		int inputsBlockSize;
		int hiddenUnitsBlockSize;
		int samplesBlockSize;

		DeviceArray<cudafloat> errors;

		RBM * previousLayer;

		DeviceArray<cudafloat> d_mse;
		DeviceAccessibleVariable<cudafloat> mse;
		CudaStream streamMSE;

		void _Init(int I, int J, int samples, cudafloat stdWeights, cudafloat initialLearningRate, cudafloat momentum, bool useBinaryValuesVisibleReconstruction, int proportionRandomValuesGenerated, RBM * previousLayer) {
			this->I = I;
			this->J = J;
			this->samples = samples;

			this->stdWeights = stdWeights;
			this->initialLearningRate = initialLearningRate;
			this->momentum = momentum;

			this->useBinaryValuesVisibleReconstruction = useBinaryValuesVisibleReconstruction;

			dimIJ.x = I;
			dimIJ.y = J;

			dimJI.x = J;
			dimJI.y = I;

			inputsBlockSize = NumberThreadsPerBlockThatBestFit(I);
			hiddenUnitsBlockSize = NumberThreadsPerBlockThatBestFit(J);
			samplesBlockSize = NumberThreadsPerBlockThatBestFit(samples);
		
			this->previousLayer = previousLayer;

			d_mse.ResizeWithoutPreservingData(1);

			*(mse.Pointer()) = CUDA_VALUE(1.0);
		}

		// Used by DBNs
		RBM(RBM * previousLayer, int hiddenUnits, cudafloat initialLearningRate, cudafloat momentum, bool useBinaryValuesVisibleReconstruction, cudafloat stdWeights, int proportionRandomValuesGenerated = 1) : mse(CUDA_VALUE(1.0)) {
			_Init(previousLayer->J, hiddenUnits, previousLayer->samples, stdWeights, initialLearningRate, momentum, useBinaryValuesVisibleReconstruction, proportionRandomValuesGenerated, previousLayer);
		}

		void ComputeStatusUnits(cudafloat * v, cudafloat * h, cudafloat * v_reconstructed, int samples, float * rnd);

	public:
		//! Initializes the RBM on the device (GPU). Must be called before performing any other operation.
		//! \return true if the initialization was successful.
		//! \sa Dispose
		bool Init(int miniBatchSize, int cd) {
		    this->miniBatchSize = miniBatchSize;
			int numberWeights = I * J;

			if (previousLayer != nullptr) {
				previousLayer->ComputeStatusUnits(previousLayer->v.Pointer(), previousLayer->h_data.Pointer(), nullptr, samples, nullptr);
				v.TransferOwnerShipFrom(previousLayer->h_data);
				previousLayer->DisposeDeviceInformation();
			}

			if (w.ResizeWithoutPreservingData(J, I) != numberWeights) return false;
			if (a.ResizeWithoutPreservingData(I) != I) return false;
			if (b.ResizeWithoutPreservingData(J) != J) return false;

			if (!lastDelta.Resize(I, J)) return false;

			if (!learningRate.Resize(I, J)) return false;
			if (!lastDeltaWithoutLearningMomentum.Resize(I, J)) return false;

			if (h_data.ResizeWithoutPreservingData(samples, J) != samples * J) return false;
			if (h_recon.ResizeWithoutPreservingData(samples, J) != h_data.Elements()) return false;
			if (v_recon.ResizeWithoutPreservingData(samples, I) != samples * I) return false;

			int randomValuesNeededPerEpoch = J;
			if (useBinaryValuesVisibleReconstruction) randomValuesNeededPerEpoch += I;
			randomValuesNeededPerEpoch *= samples;

			int numberRandomValues = randomValuesNeededPerEpoch * cd;

			if (randomValues.ResizeWithoutPreservingData(numberRandomValues) != numberRandomValues) return false;

			RandomizeWeights();

			if (errors.ResizeWithoutPreservingData(I) != I) return false;

			return true;
		}
		
		//! Disposes the information contained in the device (GPU) memory.
		//! The RBM can not be trained afterwards. 
		//! This method is called automatically if an RBM on top of this one is initialized.
		//! \sa Init
		void DisposeDeviceInformation() {
			v.Dispose();
						
			a.DisposeDevice();
			b.DisposeDevice();
			w.DisposeDevice();
		
			lastDelta.Dispose();
		
			learningRate.Dispose();
			lastDeltaWithoutLearningMomentum.Dispose(); 
		
			h_data.Dispose();
			h_recon.Dispose();
			v_recon.Dispose();

			randomValues.Dispose();
			errors.Dispose();
		}
		
		//! Constructs a Restricted Boltzman Machine that can be trained using a device (GPU).
		//! \param visibleUnits Number of inputs.
		//! \param hiddenUnits Number of hidden neurons.
		//! \param inputs Inputs of the training dataset. Each row of the matrix should contain a pattern (sample) and each column an input.
		//! \param initialLearningRate Initial learning rate.
		//! \param momentum Momentum (optional).
		//! \param useBinaryValuesVisibleReconstruction Use binary values for the visibible layer reconstruction (optional, default = false)
		//! \param stdWeights Defines the maximum and minimum value for the weights. The weights will be initialized with a random number between -stdWeights and stdWeights.
		//! \param proportionRandomValuesGenerated Proportion of random values generated (currently ignored).
		//! \sa Init
		RBM(int visibleUnits, int hiddenUnits, HostMatrix<cudafloat> & inputs, cudafloat initialLearningRate, cudafloat momentum = DEFAULT_MOMENTUM, bool useBinaryValuesVisibleReconstruction = false, cudafloat stdWeights = STD_WEIGHTS, int proportionRandomValuesGenerated = 1 /*, curandGenerator_t randomGenerator = nullptr*/)  : mse(CUDA_VALUE(1.0))  {
			assert(visibleUnits == inputs.Columns() && inputs.IsRowMajor());
			assert(proportionRandomValuesGenerated > 0);

			v = inputs;

			_Init(visibleUnits, hiddenUnits, inputs.Rows(), stdWeights, initialLearningRate, momentum, useBinaryValuesVisibleReconstruction, proportionRandomValuesGenerated, nullptr);
		}

		//! Randomizes the weights of the RBM
		void RandomizeWeights();

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

		//! Trains the RBM for one epoch using the CD-n algorithm
		//! \param n Value of n in the CD-n algorithm
		void ContrastiveDivergence(int n);

		//! Gets an estimate of the reconstruction mean square error of the network.
		//! \return An estimate of the reconstruction mean square error.
		//! \sa GetMSE
		cudafloat GetMSEestimate() {
			if (mse.Value() >= CUDA_VALUE(1.0)) return GetMSE();

			if (cudaStreamQuery(streamMSE) == cudaSuccess) {
				Reduction::Sum(errors, d_mse, 1.0f, streamMSE);
				mse.UpdateValueAsync(d_mse.Pointer(), streamMSE);
			}

			return mse.Value() / ((cudafloat) I * samples);
		}

		//! Gets an approximation the reconstruction mean square error of the network.
		//! The value given is the one of the previous epoch (hence an approximation of the current).
		//! \return An aproximation of (actual) the reconstruction mean square error.
		//! \attention Do not use during training or it will slow down considerably the training process. Use GetMSEestimate instead.
		//! \sa GetMSEestimate
		cudafloat GetMSE() {
			if (errors.Length() > 0) {
				Reduction::Sum(errors, d_mse);
				mse.UpdateValue(d_mse.Pointer());
			}

			return mse.Value() / ((cudafloat) I * samples);
		}

		//! Gets the current epoch
		//! \return the training epoch
		int Epoch() const {
			return epoch;
		}

		//! Gets the weights matrix
		//! \return the weights matrix
		HostMatrix<cudafloat> GetWeights() {
			if (w.DevicePointer() != nullptr) w.UpdateHost();
			return HostMatrix<cudafloat>(w.GetHostMatrix());
		}

		//! Gets the visible units bias
		//! \return an array with the bias
		HostArray<cudafloat> GetVisibleBias() {
			if (a.DevicePointer() != nullptr) a.UpdateHost();
			return HostArray<cudafloat>(a.GetHostArray());
		}
		
		//! Gets the hidden units bias
		//! \return an array with the bias
		HostArray<cudafloat> GetHiddenBias() {
			if (b.DevicePointer() != nullptr) b.UpdateHost();
			return HostArray<cudafloat>(b.GetHostArray());
		}

		//! Gets the number of inputs
		//! \return the number of inputs
		int GetNumberInputs() const {
			return I;
		}

		//! Gets the number of outputs (neurons in the RBM hidden layer)
		//! \return the number of outputs
		int GetNumberOutputs() const {
			return J;
		}
};

//! @}

}

#endif