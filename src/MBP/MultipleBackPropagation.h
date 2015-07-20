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

#ifndef GPUMLib_MultipleBackPropagation_h
#define GPUMLib_MultipleBackPropagation_h

#include "BackPropagation.h"

namespace GPUMLib {

//! \addtogroup mbp Multiple Back-Propagation class
//! @{

//! Represents a multiple feed-forward network that can be trained using the CUDA implementation of the Multiple Back-Propagation algorithm.
class MultipleBackPropagation : public BackPropagation {
	private:
		HostArray<bool> layerHasSelectiveNeurons;

	public:
		//! Constructs a multiple feed-forward network that can be trained using the CUDA implementation of the Multiple Back-Propagation algorithm.
		//! \param sizeLayers Number of neurons of each layer (must include the input layer) of the main network.
		//! \param selectiveNeurons Indicates which layers have selective activation neurons. At least one layer must contain selective activation neurons.
		//! \param sizeAdditionalSpaceLayers Additional layers (besides the output layer) of the space network.
		//! \param trainInputPatterns Inputs of the training dataset. Each row of the matrix should contain a pattern (sample) and each column an input.
		//! \param trainDesiredOutputPatterns Desired outputs for the training dataset. Each row of the matrix should contain a pattern (sample) and each column the desired values (labels) for an output.
		//! \param initialLearningRate The initial learning rate (step sizes) of the network.
		//! \warning You have to include the input layer in the constructor. But when calling the class methods, the first hidden layer will be 0 (zero).
		MultipleBackPropagation(HostArray<int> & sizeLayers, HostArray<bool> & selectiveNeurons, HostArray<int> & sizeAdditionalSpaceLayers, HostMatrix<cudafloat> & trainInputPatterns, HostMatrix<cudafloat> & trainDesiredOutputPatterns, cudafloat initialLearningRate = INITIAL_LEARNING_RATE);

		//! Indicates if a given layer has neurons with selective activation.
		//! \return true if the layer has neurons with selective activation, false otherwise.
		bool HasSelectiveNeurons(int layer) const;

		//! Gets the number of layers of the space network (does not include the input layer).
		//! \return The number of layers of the space network (not including the input layer).
		int GetNumberLayersSpaceNetwork() const;

		//! Gets the number of neurons of a given layer of the space network.
		//! \param layer the layer.
		//! \return The number of neurons of the space network layer.
		int GetNumberNeuronsSpaceNetwork(int layer) const;

		//! Gets the input weights of a given layer of the space network.
		//! \param layer the layer.
		//! \return The weights.
		HostArray<cudafloat> GetLayerWeightsSpaceNetwork(int layer);

		//! Sets the input weights of a given layer of the space network.
		//! \param layer the layer.
		//! \param weights The weights.
		void SetLayerWeightsSpaceNetwork(int layer, HostArray<cudafloat> & weights);

		//! Gets the selective input weights of the network.
		//! \return The weights.
		HostArray<cudafloat> GetSelectiveInputWeightsSpaceNetwork();

		//! Sets the selective input weights of the network.
		//! \param weights The weights.
		void SetSelectiveInputWeightsSpaceNetwork(HostArray<cudafloat> & weights);

		//! Gets the selective input bias of the network.
		//! \return The bias.
		HostArray<cudafloat> GetSelectiveInputBiasSpaceNetwork();

		//! Sets the selective input bias of the network.
		//! \param bias The bias.
		void SetSelectiveInputBiasSpaceNetwork(HostArray<cudafloat> & bias);
};

//! \example MBP.cpp 
//! Example of the CUDA Multiple Back-Propagation algorithm usage (two-spirals benchmark).
//! \example ATS.cpp
//! Example Autonomous Training System (uses the Back-Propagation and Multiple Back-Propagation algorithms).

//! @}

}

#endif