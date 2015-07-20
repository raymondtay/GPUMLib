/*
	Noel Lopes is an Assistant Professor at the Polytechnic Institute of Guarda, Portugal
	Copyright (C) 2009, 2010, 2011, 2012, 2013, 2014 Noel de Jesus Mendonça Lopes

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

#ifndef GPUMLib_DBN_h
#define GPUMLib_DBN_h

#include "../RBM/RBM.h"
#include "../reduction/reduction.h"

namespace GPUMLib {

//! \addtogroup dbn Deep Belief Networks device (GPU) class
//! @{

//! Represents a Deep Belief Network (Device - GPU).
class DBN {
	private:
		HostArray<RBM *> rbms;

	public:
		//! Constructs a Deep Belief Network that can be trained using the CPU (Host).
		//! \param layers Number of units in each layer.
		//! \param inputs Inputs of the training dataset. Each row of the matrix should contain a pattern (sample) and each column an input.
		//! \param initialLearningRate Initial learning rate.
		//! \param momentum Momentum
		//! \param useBinaryValuesVisibleReconstruction Use binary values for the visibible layer reconstruction
		//! \param stdWeights Defines the maximum and minimum value for the weights. The weights will be initialized with a random number between -stdWeights and stdWeights.
		DBN(HostArray<int> & layers, HostMatrix<cudafloat> & inputs, cudafloat initialLearningRate, cudafloat momentum = DEFAULT_MOMENTUM, bool useBinaryValuesVisibleReconstruction = false, cudafloat stdWeights = STD_WEIGHTS) {
			int nlayers = layers.Length();
			assert(nlayers >= 2);

			rbms.ResizeWithoutPreservingData(nlayers - 1);

			rbms[0] = new RBM(layers[0], layers[1], inputs, initialLearningRate, momentum, useBinaryValuesVisibleReconstruction, stdWeights);

			for(int r = 1; r < nlayers - 1; r++) {
				rbms[r] = new RBM(rbms[r - 1], layers[r + 1], initialLearningRate, momentum, useBinaryValuesVisibleReconstruction, stdWeights);
			}
		}

		~DBN() {
			int nlayers = rbms.Length();

			for(int l = 0; l < nlayers; l++) {
				delete rbms[l];
				rbms[l] = nullptr;
			}
		}

		////! Randomizes the weights of the RBM, between -stdWeights and stdWeights.
		////! \param stdWeights Defines the maximum and minimum value for the weights.
		////void RandomizeWeights(cudafloat stdWeights, cudafloat initialLearningRate) {
		//	//int nlayers = rbms.Length();
		//	//for(int l = 0; l < nlayers; l++) rbms[l]->RandomizeWeights(stdWeights, initialLearningRate);
		////}

		//! Get an RBM that is part of the DBN.
		//! \param layer Layer to obtain.
		//! \return The RBM corresponding to layer specified.
		RBM * GetRBM(int layer) {
			assert(layer < rbms.Length());
			return rbms[layer];
		}

		//! Gets the number of RBMs.
		//! \return The number of RBMs that compose the DBN.
		int GetNumberRBMs() const {
			return rbms.Length();
		}

		//! Train the DBN (train each RBM).
		//! \param epochs Maximum number of epochs that each RBM should be trained.
		//! \param cd Define the value of k in CD-k.
		//! \param miniBatchSize Mini Batch Size.
		//! \param errorStop Stop the training in each RBM when the Mean Square Error (MSE) is inferior to this value.
		//! \return True if successful. False otherwise.
		bool Train(int epochs, int cd, int miniBatchSize, cudafloat errorStop = CUDA_VALUE(0.0)) {
			RBM * layer;

			int nlayers = rbms.Length();
			for(int l = 0; l < nlayers; l++) {
				layer = rbms[l];
				if (!layer->Init(miniBatchSize, cd)) return false;

				for (int e = 0; e < epochs; e++) {
					layer->ContrastiveDivergence(cd);
					if (errorStop != CUDA_VALUE(0.0) && layer->GetMSEestimate() < errorStop) break;
				}
			}

			layer->DisposeDeviceInformation();

			return true;
		}
};

//! \example DBNapp.cpp 
//! Example of the DBN and RBM algorithms usage.

//! @}

}

#endif