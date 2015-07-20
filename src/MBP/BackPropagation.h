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

#ifndef BackPropagation_h
#define BackPropagation_h

#define INITIAL_LEARNING_RATE (CUDA_VALUE(0.7))

#include "../common/CudaDefinitions.h"
#include "../common/CudaStreams.h"
#include "../memory/DeviceArray.h"
#include "../memory/DeviceMatrix.h"
#include "../memory/DeviceAccessibleVariable.h"

namespace GPUMLib {

//! \addtogroup bp Back-Propagation class
//! @{

//! Represents a feed-forward network that can be trained using the CUDA implementation of the Back-Propagation algorithm.
class BackPropagation {
	private:
		class Layer {
			friend class BackPropagation;
			friend class MultipleBackPropagation;

			private:
				static int totalNeuronsWithSelectiveActivation;
				static int patterns;

				DeviceArray<cudafloat> d_weights;
				DeviceArray<cudafloat> d_bestWeights;
				DeviceArray<cudafloat> d_learnRate;
				DeviceArray<cudafloat> d_lastDelta;
				DeviceArray<cudafloat> d_lastDeltaWithoutLearningMomentum;
				DeviceArray<cudafloat> d_outputs;
				DeviceArray<cudafloat> d_localGradient;

				int neurons;
				int connections;
				int inputsWithoutBias;
				int mOffset;

				cudafloat * d_inputs;
				cudafloat * d_desOutputs;
				cudafloat * d_m;
				cudafloat * d_localGradSpaceNet;
				cudafloat * d_rms;

				dim3 dimInputsNeurons;
				dim3 dimOutputsNeurons;

				int inputsBlockSize;
				int sharedMemFire;
				int sharedMemGradients;

				bool isOutputLayer;

				void RandomizeWeights(cudafloat minValue, cudafloat maxValue, cudafloat initialLearningRate);
				void Init(int neurons, int inputs, int nextLayerNeurons, cudafloat initialLearningRate, cudafloat * layerInputs, bool isOutputLayer, cudafloat * m = nullptr, cudafloat * localGradSpaceNet = nullptr, int mOffset = 0);
			
				void Fire(cudaStream_t stream);

				void CalculateLocalGradient(cudaStream_t stream, cudafloat * rms, cudafloat * bestRMS, cudafloat rmsGrowToApplyRobustLearning, Layer & nextLayer);

				void CorrectWeights(cudaStream_t stream, int patternsBlockSize, cudafloat * rms, cudafloat * bestRMS, cudafloat rmsGrowToApplyRobustLearning, cudafloat robustFactor, cudafloat momentum, cudafloat u, cudafloat d, cudafloat maxStepSize);

				void CorrectWeights(cudaStream_t stream, int patternsBlockSize, cudafloat * rms, cudafloat * bestRMS, cudafloat rmsGrowToApplyRobustLearning, cudafloat robustFactor, cudafloat momentum, cudafloat u, cudafloat d);
		};

		class SelectiveInputLayer {
			friend class BackPropagation;
			friend class MultipleBackPropagation;

			private:
				int patterns;
				int neurons;

				DeviceArray<cudafloat> d_weights;
				DeviceArray<cudafloat> d_bias;
				DeviceArray<cudafloat> d_bestWeights;
				DeviceArray<cudafloat> d_bestBias;
				DeviceArray<cudafloat> d_learnRate;
				DeviceArray<cudafloat> d_learnRateBias;
				DeviceArray<cudafloat> d_lastDelta;
				DeviceArray<cudafloat> d_lastDeltaBias;
				DeviceArray<cudafloat> d_lastDeltaWithoutLearningMomentum;
				DeviceArray<cudafloat> d_lastDeltaWithoutLearningMomentumBias;
				DeviceArray<cudafloat> d_outputs;
				DeviceArray<cudafloat> d_localGradient;

				cudafloat * d_inputs;

				dim3 dimOutputsNeurons;

				int fireBlockSize;
				int fireBlocks;

				int sharedMemGradients;

				void RandomizeWeights(cudafloat minValue, cudafloat maxValue, cudafloat initialLearningRate, HostArray<bool> & selectiveInputs);
				
				SelectiveInputLayer(int patterns, HostArray<bool> & selectiveInputs, int nextLayerNeurons, cudafloat * inputs, cudafloat initialLearningRate) : 
					d_outputs(patterns * selectiveInputs.Length()), 
					dimOutputsNeurons(nextLayerNeurons, selectiveInputs.Length()), 
					d_localGradient(selectiveInputs.Length() * patterns)
				{
					this->patterns = patterns;
					this->neurons = selectiveInputs.Length();
					
					sharedMemGradients = (nextLayerNeurons * (neurons + 1)) * sizeof(cudafloat);

					this->d_inputs = inputs;

					RandomizeWeights(CUDA_VALUE(-1.0), CUDA_VALUE(1.0), initialLearningRate, selectiveInputs);
				}

				void Fire(cudaStream_t stream);
				void CalculateLocalGradient(cudaStream_t stream, cudafloat * rms, cudafloat * bestRMS, cudafloat rmsGrowToApplyRobustLearning, Layer & nextLayer);

				void CorrectWeights(cudaStream_t stream, cudafloat * rms, cudafloat * bestRMS, cudafloat rmsGrowToApplyRobustLearning, cudafloat robustFactor, cudafloat momentum, cudafloat u, cudafloat d, cudafloat maxStepSize);
		};

		DeviceMatrix<cudafloat> d_inputs;
		DeviceMatrix<cudafloat> d_desOutputs;

		int maxNumberWeigths;

		cudafloat initialLearningRate;
		cudafloat momentum;
		cudafloat u;
		cudafloat d;
		cudafloat maxStepSize;

		DeviceArray<cudafloat> d_rms;
		DeviceArray<cudafloat> d_bestRMS;
		DeviceArray<cudafloat> d_rmsOut;
		DeviceAccessibleVariable<cudafloat> rms;

		CudaStream streamKernels;
		CudaStream streamRMS;

		int patternsBlockSize;
		cudafloat numberPatternsNeurons;

		int epoch;

		// Robust learning
		DeviceArray<int> d_numberWeightsLayer;

		DeviceArray<cudafloat *> d_weightsLayers;
		DeviceArray<cudafloat *> d_bestWeightsLayers;
		DeviceArray<cudafloat *> d_learnRatesLayers;
		DeviceArray<cudafloat *> d_lastDeltaLayers;
		DeviceArray<cudafloat *> d_lastDeltaWithoutLMlayers;

		bool robustLearning;		
		int layersRobustTraining;
		cudafloat rmsGrowToApplyRobustLearning;
		cudafloat robustFactor;

		HostArray<Layer> layers;
		SelectiveInputLayer * selectiveInputLayer;

		HostArray<bool> selectiveInputs;

		void Fire();

	protected:
		HostArray<Layer> spaceLayers;
		SelectiveInputLayer * selectiveInputLayerSpaceNetwork;

		void CreateNetwork(HostArray<int> & sizeLayers, HostArray<int> * sizeSpaceLayers, HostArray<bool> * selectiveNeurons, HostMatrix<cudafloat> & trainInputPatterns, HostMatrix<cudafloat> & trainDesiredOutputPatterns, cudafloat initialLearningRate);
		BackPropagation() {}

	public:
		//! Constructs a feed-forward network that can be trained using the CUDA implementation of the Back-Propagation algorithm.
		//! \param sizeLayers Number of neurons of each layer (must include the input layer).
		//! \param trainInputPatterns Inputs of the training dataset. Each row of the matrix should contain a pattern (sample) and each column an input.
		//! \param trainDesiredOutputPatterns Desired outputs for the training dataset. Each row of the matrix should contain a pattern (sample) and each column the desired values (labels) for an output.
		//! \param initialLearningRate The initial learning rate (step sizes) of the network.
		//! \warning You have to include the input layer in the constructor. But when calling the class methods, the first hidden layer will have index 0 (zero).
		BackPropagation(HostArray<int> & sizeLayers, HostMatrix<cudafloat> & trainInputPatterns, HostMatrix<cudafloat> & trainDesiredOutputPatterns, cudafloat initialLearningRate = INITIAL_LEARNING_RATE);

		//! Destructor.
		~BackPropagation() {
			if (selectiveInputLayerSpaceNetwork != nullptr) delete selectiveInputLayerSpaceNetwork;
			if (selectiveInputLayer != nullptr) delete selectiveInputLayer;
		}

		//! Randomizes the weights of the network, between two values.
		//! \param minValue minimum value.
		//! \param maxValue maximum value
		void RandomizeWeights(cudafloat minValue, cudafloat maxValue);

		//! Gets whether robust the robust training is used during the training.
		//! \return Whether robust training is used or not.
		bool GetRobustLearning() const;

		//! Sets if robust training is used during the training.
		//! \param value true if robust learning should be used. False otherwise.
		void SetRobustLearning(bool value);

		//! Gets the maximum allowable growth of the root mean square error in terms of percentage, before the robust training is applied.
		//! \return the maximum growth percentage (e.g. 0.1% -> 0.001).
		cudafloat GetMaxPercentageRMSGrow() const;

		//! Sets the maximum allowable growth of the root mean square error in terms of percentage, before the robust training is applied.
		//! \param value the maximum grow percentage (e.g. 0.1% -> 0.001).
		void SetMaxPercentageRMSGrow(cudafloat value);

		//! Gets the robust training (reducing/decreasing) factor.
		//! \return the robust training factor.
		cudafloat GetRobustFactor() const;

		//! Sets the robust training (reducing/decreasing) factor.
		//! \param value Robust training factor.
		void SetRobustFactor(cudafloat value);

		//! Gets the momentum.
		//! \return The momentum.
		cudafloat GetMomentum() const;

		//! Sets the momentum.
		//! \param value The momentum factor.
		void SetMomentum(cudafloat value);

		//! Gets the increment (up) factor.
		//! \return The increase (up) factor.
		cudafloat GetUpStepSizeFactor() const;

		//! Sets the increment (up) factor.
		//! \param value The increment (up) factor.
		void SetUpStepSizeFactor(cudafloat value);

		//! Gets the decrement (down) factor.
		//! \return The decrement (down) factor.
		cudafloat GetDownStepSizeFactor() const;

		//! Sets the decrement (down) factor.
		//! \param value The decrement (down) factor.
		void SetDownStepSizeFactor(cudafloat value);

		//! Gets the maximum step size.
		//! \return The maximum step size.
		cudafloat GetMaxStepSize() const;

		//! Sets the maximum step size.
		//! \param value The maximum step size.
		void SetMaxStepSize(cudafloat value);

		//! Gets the current training epoch.
		//! \return The current epoch.
		int GetEpoch() const;

		//! Gets the number of layers (does not include the input layer).
		//! \return The number of layers (not including the input layer).
		int GetNumberLayers() const;

		//! Gets the number of inputs of the network.
		//! \return The number of inputs of the network
		int GetNumberInputs() const;

		//! Gets the number of outputs of the network.
		//! \return The number of outputs of the network
		int GetNumberOutputs() const;

		//! Gets the number of neurons of a given layer.
		//! \param layer the layer.
		//! \return The number of neurons of the network layer.
		int GetNumberNeurons(int layer) const;

		//! Gets an estimate of the root mean square error of the network.
		//! \return An estimate of the root mean square error.
		//! \sa GetRMS
		cudafloat GetRMSestimate();

		//! Gets the root mean square error of the network.
		//! \return The root mean square error.
		//! \attention Do not use during training or it will slow down considerably the training process. Use GetRMSestimate instead.
		//! \sa GetRMSestimate
		cudafloat GetRMS();

		//! Trains the network one epoch.
		void TrainOneEpoch();

		//! Trains the network for a given number of epochs.
		//! \param epochs number of epochs.
		void Train(int epochs);

		//! Trains the network for a given number of epochs or until its error is lower or equal than a specified RMS error.
		//! \param epochs number of epochs.
		//! \param rmsStop desired RMS error.
		void Train(int epochs, cudafloat rmsStop);

		//! Computes the network outputs for a given set of inputs.
		//! \param inputs Inputs to be presented to the network.
		//! \return A matrix containing the network outputs.
		HostMatrix<cudafloat> GetOutputs(HostMatrix<cudafloat> & inputs);

		//! Gets the input weights of a given layer of the network (includes the bias).
		//! \param layer the layer.
		//! \return The weights.
		HostArray<cudafloat> GetLayerWeights(int layer);

		//! Sets the input weights of a given layer of the network (must include the bias).
		//! \param layer the layer.
		//! \param weights The weights.
		void SetLayerWeights(int layer, HostArray<cudafloat> & weights);

		//! Sets the input weights of a given layer of the network.
		//! \param layer the layer.
		//! \param weights The weights.
		//! \param bias The bias.
		void SetLayerWeights(int layer, HostMatrix<cudafloat> & weights, HostArray<cudafloat> & bias);

		//! Sets Indicates if the network contains selective inputs.
		//! \return true if the network contains selective inputs, false otherwise
		bool HasSelectiveInputs() const {
			return (selectiveInputLayer != nullptr);
		}

		//! Gets the selective input weights of the network.
		//! \return The weights.
		HostArray<cudafloat> GetSelectiveInputWeights();

		//! Sets the selective input weights of the network.
		//! \param weights The weights.
		void SetSelectiveInputWeights(HostArray<cudafloat> & weights);

		//! Gets the selective input bias of the network.
		//! \return The bias.
		HostArray<cudafloat> GetSelectiveInputBias();

		//! Sets the selective input bias of the network.
		//! \param bias The bias.
		void SetSelectiveInputBias(HostArray<cudafloat> & bias);
};

//! \example BP.cpp 
//! Example of the CUDA Back-Propagation algorithm usage (two-spirals benchmark).
//! \example ATS.cpp
//! Example Autonomous Training System (uses the Back-Propagation and Multiple Back-Propagation algorithms).

//! @}

}

#endif