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

#include <assert.h>
#include <stdlib.h>
#include "BackPropagation.h"
#include "MBPkernels.h"

namespace GPUMLib {

int BackPropagation::Layer::totalNeuronsWithSelectiveActivation = 0;
int BackPropagation::Layer::patterns;

void BackPropagation::Layer::RandomizeWeights(cudafloat minValue, cudafloat maxValue, cudafloat initialLearningRate) {
	assert(maxValue > minValue);

	HostArray<cudafloat> learnRate(connections);
	HostArray<cudafloat> delta(connections);
	HostArray<cudafloat> weights(connections);

	for(int c = 0; c < connections; c++) {
		weights[c] = (maxValue - minValue) * ((cudafloat) rand() / RAND_MAX) + minValue;
		learnRate[c] = initialLearningRate;
		delta[c] = CUDA_VALUE(0.0);
	}

	d_bestWeights = d_weights = weights;
	d_learnRate = learnRate;
	d_lastDelta = d_lastDeltaWithoutLearningMomentum = delta;
}

void BackPropagation::Layer::Fire(cudaStream_t stream) {
	dim3 dimNeuronsPatterns;
	dimNeuronsPatterns.x = neurons;

	if (isOutputLayer) {
		if(connections > MAX_THREADS_PER_BLOCK) {
			int processed = 0;
			do {
				int patternsToProcess = (patterns > 65535) ? 65535 : patterns;				
				dimNeuronsPatterns.y = patternsToProcess;
				KernelFireOutputLayer(stream, dimNeuronsPatterns, inputsBlockSize, d_inputs + (processed * inputsWithoutBias), d_weights.Pointer(), (d_m != nullptr) ? d_m + (processed * totalNeuronsWithSelectiveActivation) : nullptr, mOffset, totalNeuronsWithSelectiveActivation, d_desOutputs + (processed * neurons), d_outputs.Pointer() + (processed * neurons), d_localGradient.Pointer() + (processed * neurons), d_rms + processed, (d_localGradSpaceNet == nullptr) ? nullptr : d_localGradSpaceNet + (processed * totalNeuronsWithSelectiveActivation), inputsWithoutBias);
				processed += patternsToProcess;
			} while (processed < patterns);			
		} else {
			int processed = 0;
			do {
				int patternsToProcess = (patterns > 65535) ? 65535 : patterns;
				FireOutputLayer<<<patternsToProcess, dimInputsNeurons, sharedMemFire, stream>>>(d_inputs + (processed * inputsWithoutBias), d_weights.Pointer(), (d_m == nullptr) ? nullptr : d_m + (processed * totalNeuronsWithSelectiveActivation), mOffset, totalNeuronsWithSelectiveActivation, d_desOutputs + (processed * neurons), d_outputs.Pointer() + (processed * neurons), d_localGradient.Pointer() + (processed * neurons), d_rms + processed, (d_m == nullptr) ? nullptr : d_localGradSpaceNet + (processed * totalNeuronsWithSelectiveActivation));
				processed += patternsToProcess;
			} while (processed < patterns);
		}
	} else {
		if(connections > MAX_THREADS_PER_BLOCK) {
			int processed = 0;
			do {
				int patternsToProcess = (patterns > 65535) ? 65535 : patterns;				
				dimNeuronsPatterns.y = patternsToProcess;
				KernelFireLayer(stream, dimNeuronsPatterns, inputsBlockSize, d_inputs + (processed * inputsWithoutBias), d_weights.Pointer(), (d_m != nullptr) ? d_m + (processed * totalNeuronsWithSelectiveActivation) : nullptr, mOffset, totalNeuronsWithSelectiveActivation, d_outputs.Pointer() + (processed * neurons), inputsWithoutBias);
				processed += patternsToProcess;
			} while (processed < patterns);
		} else {
			int processed = 0;
			do {
				int patternsToProcess = (patterns > 65535) ? 65535 : patterns;
				FireLayer<<<patternsToProcess, dimInputsNeurons, sharedMemFire, stream>>>(d_inputs + (processed * inputsWithoutBias), d_weights.Pointer(), (d_m != nullptr) ? d_m + (processed * totalNeuronsWithSelectiveActivation) : nullptr, mOffset, totalNeuronsWithSelectiveActivation, d_outputs.Pointer() + (processed * neurons));
				processed += patternsToProcess;
			} while (processed < patterns);
		}
	}
}

void BackPropagation::Layer::CalculateLocalGradient(cudaStream_t stream, cudafloat * rms, cudafloat * bestRMS, cudafloat rmsGrowToApplyRobustLearning, Layer & nextLayer) {
	int processed = 0;
	do {
		int patternsToProcess = (patterns > 65535) ? 65535 : patterns;
		GPUMLib::CalculateLocalGradient<<<patternsToProcess, dimOutputsNeurons, sharedMemGradients, stream>>>(rms, bestRMS, rmsGrowToApplyRobustLearning, d_outputs.Pointer() + (processed * neurons), nextLayer.d_weights.Pointer(), (d_m != nullptr) ? d_m + (processed * totalNeuronsWithSelectiveActivation) : nullptr, mOffset, totalNeuronsWithSelectiveActivation, nextLayer.d_localGradient.Pointer() + (processed * dimOutputsNeurons.x), d_localGradient.Pointer() + (processed * neurons), (d_m == nullptr) ? nullptr : d_localGradSpaceNet + (processed * totalNeuronsWithSelectiveActivation));
		processed += patternsToProcess;
	} while (processed < patterns);
}

void BackPropagation::Layer::CorrectWeights(cudaStream_t stream, int patternsBlockSize, cudafloat * rms, cudafloat * bestRMS, cudafloat rmsGrowToApplyRobustLearning, cudafloat robustFactor, cudafloat momentum, cudafloat u, cudafloat d, cudafloat maxStepSize) {
	KernelCorrectLayerWeights(stream, dimInputsNeurons, patternsBlockSize, rms, bestRMS, rmsGrowToApplyRobustLearning, d_inputs, d_localGradient.Pointer(), d_weights.Pointer(), d_learnRate.Pointer(), d_lastDeltaWithoutLearningMomentum.Pointer(), d_lastDelta.Pointer(), maxStepSize, u, d, robustFactor, momentum, patterns);
}

void BackPropagation::Layer::Init(int neurons, int inputs, int nextLayerNeurons, cudafloat initialLearningRate, cudafloat * layerInputs, bool isOutputLayer, cudafloat * m, cudafloat * localGradSpaceNet, int mOffset) {
	connections = inputs * neurons;

	this->neurons = neurons;	
	inputsWithoutBias = inputs - 1;

	RandomizeWeights(CUDA_VALUE(-1.0), CUDA_VALUE(1.0), initialLearningRate);

	d_m = m;
	d_localGradSpaceNet = localGradSpaceNet;
	this->mOffset = mOffset;

	inputsBlockSize = 1;
	while(inputsBlockSize < MAX_THREADS_PER_BLOCK && inputsBlockSize < inputs) inputsBlockSize <<= 1;

	d_inputs = layerInputs;
	d_outputs.ResizeWithoutPreservingData(neurons * patterns);
	d_localGradient.ResizeWithoutPreservingData(neurons * patterns);
		
	sharedMemFire = connections * sizeof(cudafloat);
	sharedMemGradients = (nextLayerNeurons * (neurons + 1)) * sizeof(cudafloat);

	dimInputsNeurons.x = inputs;
	dimInputsNeurons.y = neurons;

	dimOutputsNeurons.x = nextLayerNeurons;
	dimOutputsNeurons.y = neurons;

	this->isOutputLayer = isOutputLayer;
}

void BackPropagation::SelectiveInputLayer::RandomizeWeights(cudafloat minValue, cudafloat maxValue, cudafloat initialLearningRate, HostArray<bool> & selectiveInputs) {
	assert(maxValue > minValue);

	int ninputs = selectiveInputs.Length();

	HostArray<cudafloat> weights(ninputs);
	HostArray<cudafloat> bias(ninputs);
	HostArray<cudafloat> learningRate(ninputs);
	HostArray<cudafloat> delta(ninputs);

	for(int i = 0; i < ninputs; i++) {
		if (selectiveInputs[i]) {
			weights[i] = (maxValue - minValue) * ((cudafloat) rand() / RAND_MAX) + minValue;
			bias[i] = (maxValue - minValue) * ((cudafloat) rand() / RAND_MAX) + minValue;
		} else {
			weights[i] = CUDA_VALUE(0.0);
			bias[i] = CUDA_VALUE(0.0);
		}

		learningRate[i] = initialLearningRate;
		delta[i] = CUDA_VALUE(0.0);
	}

	d_bestWeights = d_weights = weights;
	d_bestBias = d_bias = bias;
	d_learnRateBias = d_learnRate = learningRate;
	d_lastDelta = d_lastDeltaBias = d_lastDeltaWithoutLearningMomentum = d_lastDeltaWithoutLearningMomentumBias = delta;
}

void BackPropagation::SelectiveInputLayer::Fire(cudaStream_t stream) {
	int processed = 0;
	do {
		int patternsToProcess = (patterns > 65535) ? 65535 : patterns;
		FireSelectiveInputs<<<patternsToProcess, neurons, 0, stream>>>(d_inputs + (processed * neurons), d_weights.Pointer(), d_bias.Pointer(), d_outputs.Pointer() + (processed * neurons), neurons);
		processed += patternsToProcess;
	} while (processed < patterns);
}

void BackPropagation::SelectiveInputLayer::CalculateLocalGradient(cudaStream_t stream, cudafloat * rms, cudafloat * bestRMS, cudafloat rmsGrowToApplyRobustLearning, Layer & nextLayer) {
	int processed = 0;
	do {
		int patternsToProcess = (patterns > 65535) ? 65535 : patterns;
		CalcLocalGradSelectiveInputs<<<patternsToProcess, dimOutputsNeurons, sharedMemGradients, stream>>>(rms, bestRMS, rmsGrowToApplyRobustLearning, d_inputs + (processed * neurons), d_weights.Pointer(), d_bias.Pointer(), nextLayer.d_weights.Pointer(), nextLayer.d_localGradient.Pointer() + (processed * dimOutputsNeurons.x), d_localGradient.Pointer() + (processed * neurons));
		processed += patternsToProcess;
	} while (processed < patterns);
}

void BackPropagation::SelectiveInputLayer::CorrectWeights(cudaStream_t stream, cudafloat * rms, cudafloat * bestRMS, cudafloat rmsGrowToApplyRobustLearning, cudafloat robustFactor, cudafloat momentum, cudafloat u, cudafloat d, cudafloat maxStepSize) {
	KernelCorrectWeightsSelectiveInputs(stream, neurons, patterns, rms, bestRMS, rmsGrowToApplyRobustLearning, d_inputs, d_localGradient.Pointer(), d_weights.Pointer(), d_bias.Pointer(), d_learnRate.Pointer(), d_learnRateBias.Pointer(), d_lastDeltaWithoutLearningMomentum.Pointer(), d_lastDeltaWithoutLearningMomentumBias.Pointer(), d_lastDelta.Pointer(), d_lastDeltaBias.Pointer(), u, d, maxStepSize, robustFactor, momentum, patterns);
}

void BackPropagation::CreateNetwork(HostArray<int> & sizeLayers, HostArray<int> * sizeSpaceLayers, HostArray<bool> * selectiveNeurons, HostMatrix<cudafloat> & trainInputPatterns, HostMatrix<cudafloat> & trainDesiredOutputPatterns, cudafloat initialLearningRate) {
	int nsamples = trainInputPatterns.Rows();
	int ninputs = trainInputPatterns.Columns();

	Layer::patterns = nsamples;
	assert(Layer::patterns > 0 && Layer::patterns == trainDesiredOutputPatterns.Rows());

	d_inputs = trainInputPatterns;
	d_desOutputs = trainDesiredOutputPatterns;

	d_rmsOut.ResizeWithoutPreservingData(1);

	this->initialLearningRate = initialLearningRate;
	assert(initialLearningRate > CUDA_VALUE(0.0));

	// Check for selective inputs
	bool hasSelectiveInputs = false;
	
	selectiveInputs.ResizeWithoutPreservingData(ninputs);
	for(int i = 0; i < ninputs; i++) selectiveInputs[i] = false;

	int fi = 0;
	int li = ninputs - 1;

	for(int s = 0; s < nsamples; s++) {
		for(int i = fi; i <= li; i++) {
			if (!selectiveInputs[i] && IsInfOrNaN(trainInputPatterns(s, i))) {
				selectiveInputs[i] = hasSelectiveInputs = true;
				if (i == fi) fi++; else if (i == li) li--;
			}
		}

		if (fi >= li) break;
	}

	//Create the space layers
	int numberSpaceLayers = (sizeSpaceLayers == nullptr) ? 0 : sizeSpaceLayers->Length();

	selectiveInputLayerSpaceNetwork = nullptr;
	
	if (numberSpaceLayers) {
		assert(selectiveNeurons != nullptr);

		spaceLayers.ResizeWithoutPreservingData(numberSpaceLayers);

		int inputsWithoutBias = sizeLayers[0];

		cudafloat * layerInputs = d_inputs.Pointer();

		if (hasSelectiveInputs) {
			selectiveInputLayerSpaceNetwork = new SelectiveInputLayer(nsamples, selectiveInputs, (*sizeSpaceLayers)[0], layerInputs, initialLearningRate);
			layerInputs = selectiveInputLayerSpaceNetwork->d_outputs.Pointer();
		}

		for(int l = 0; l < numberSpaceLayers; l++) {
			int neurons = (*sizeSpaceLayers)[l];

			int nextLayerNeurons;
			
			if (l == numberSpaceLayers - 1) {
				Layer::totalNeuronsWithSelectiveActivation = neurons;
				nextLayerNeurons = 0;
			} else {
				nextLayerNeurons = (*sizeSpaceLayers)[l + 1];
			}

			spaceLayers[l].Init(neurons, inputsWithoutBias + 1, nextLayerNeurons, initialLearningRate, layerInputs, false);

			layerInputs = spaceLayers[l].d_outputs.Pointer();
			inputsWithoutBias = neurons;
		}
	}

	//Create the layers
	int numberLayers = sizeLayers.Length() - 1;
	assert(numberLayers > 0);

	layers.ResizeWithoutPreservingData(numberLayers);

	int outputLayer = numberLayers - 1;

	int inputsWithoutBias = sizeLayers[0];
	assert(inputsWithoutBias > 0 && inputsWithoutBias == trainInputPatterns.Columns());

	cudafloat * layerInputs = d_inputs.Pointer();

	if (hasSelectiveInputs) {
		selectiveInputLayer = new SelectiveInputLayer(nsamples, selectiveInputs, sizeLayers[1], layerInputs, initialLearningRate);
		layerInputs = selectiveInputLayer->d_outputs.Pointer();
	} else {
		selectiveInputLayer = nullptr;
	}

	cudafloat * m = (numberSpaceLayers == 0) ? nullptr : spaceLayers[numberSpaceLayers - 1].d_outputs.Pointer();
	cudafloat * localGradSpaceNet = (numberSpaceLayers == 0) ? nullptr : spaceLayers[numberSpaceLayers - 1].d_localGradient.Pointer(); 
	int mOffset = 0;

	for(int l = 0; l < numberLayers; l++) {
		int neurons = sizeLayers[l + 1];
		assert(neurons > 0);

		bool isOutputLayer = (l == outputLayer) ? true : false;

		int nextLayerNeurons = (isOutputLayer) ? 0 : sizeLayers[l + 2];

		bool hasSelectiveNeurons = (numberSpaceLayers > 0 && (*selectiveNeurons)[l]) ? true : false;

		layers[l].Init(neurons, inputsWithoutBias + 1, nextLayerNeurons, initialLearningRate, layerInputs, isOutputLayer, (hasSelectiveNeurons) ? m : nullptr, (hasSelectiveNeurons) ? localGradSpaceNet : nullptr, mOffset);

		if (hasSelectiveNeurons) mOffset += neurons;

		layerInputs = layers[l].d_outputs.Pointer();
		inputsWithoutBias = neurons;
	}

	//Robust Learning
	layersRobustTraining = numberLayers + numberSpaceLayers;
	if (hasSelectiveInputs) layersRobustTraining += (numberSpaceLayers) ? 4 : 2;

	HostArray<int> numberWeightsLayer(layersRobustTraining);
	HostArray<cudafloat *> weightsLayers(layersRobustTraining);
	HostArray<cudafloat *> bestWeightsLayers(layersRobustTraining);
	HostArray<cudafloat *> learnRatesLayers(layersRobustTraining);
	HostArray<cudafloat *> lastDeltaLayers(layersRobustTraining);
	HostArray<cudafloat *> lastDeltaWithoutLMlayers(layersRobustTraining);

	maxNumberWeigths = 0;

	int ll = 0;
	while (ll < numberSpaceLayers) {
		int connections = spaceLayers[ll].connections;
		if (connections > maxNumberWeigths) maxNumberWeigths = connections;

		numberWeightsLayer[ll] = connections;
		weightsLayers[ll] = spaceLayers[ll].d_weights.Pointer();
		bestWeightsLayers[ll] = spaceLayers[ll].d_bestWeights.Pointer();
		learnRatesLayers[ll] = spaceLayers[ll].d_learnRate.Pointer();
		lastDeltaLayers[ll] = spaceLayers[ll].d_lastDelta.Pointer();
		lastDeltaWithoutLMlayers[ll] = spaceLayers[ll].d_lastDeltaWithoutLearningMomentum.Pointer();

		ll++;
	}

	for(int l = 0; l < numberLayers; l++) {
		int connections = layers[l].connections;
		if (connections > maxNumberWeigths) maxNumberWeigths = connections;

		numberWeightsLayer[ll] = connections;
		weightsLayers[ll] = layers[l].d_weights.Pointer();
		bestWeightsLayers[ll] = layers[l].d_bestWeights.Pointer();
		learnRatesLayers[ll] = layers[l].d_learnRate.Pointer();
		lastDeltaLayers[ll] = layers[l].d_lastDelta.Pointer();
		lastDeltaWithoutLMlayers[ll] = layers[l].d_lastDeltaWithoutLearningMomentum.Pointer();

		ll++;
	}

	if (hasSelectiveInputs) {
		numberWeightsLayer[ll] = ninputs;
		weightsLayers[ll] = selectiveInputLayer->d_weights.Pointer();
		bestWeightsLayers[ll] = selectiveInputLayer->d_bestWeights.Pointer();
		learnRatesLayers[ll] = selectiveInputLayer->d_learnRate.Pointer();
		lastDeltaLayers[ll] = selectiveInputLayer->d_lastDelta.Pointer();
		lastDeltaWithoutLMlayers[ll] = selectiveInputLayer->d_lastDeltaWithoutLearningMomentum.Pointer();
		ll++;

		numberWeightsLayer[ll] = ninputs;
		weightsLayers[ll] = selectiveInputLayer->d_bias.Pointer();
		bestWeightsLayers[ll] = selectiveInputLayer->d_bestBias.Pointer();
		learnRatesLayers[ll] = selectiveInputLayer->d_learnRateBias.Pointer();
		lastDeltaLayers[ll] = selectiveInputLayer->d_lastDeltaBias.Pointer();
		lastDeltaWithoutLMlayers[ll] = selectiveInputLayer->d_lastDeltaWithoutLearningMomentumBias.Pointer();
		ll++;

		if (numberSpaceLayers) {
			numberWeightsLayer[ll] = ninputs;
			weightsLayers[ll] = selectiveInputLayerSpaceNetwork->d_weights.Pointer();
			bestWeightsLayers[ll] = selectiveInputLayerSpaceNetwork->d_bestWeights.Pointer();
			learnRatesLayers[ll] = selectiveInputLayerSpaceNetwork->d_learnRate.Pointer();
			lastDeltaLayers[ll] = selectiveInputLayerSpaceNetwork->d_lastDelta.Pointer();
			lastDeltaWithoutLMlayers[ll] = selectiveInputLayerSpaceNetwork->d_lastDeltaWithoutLearningMomentum.Pointer();
			ll++;

			numberWeightsLayer[ll] = ninputs;
			weightsLayers[ll] = selectiveInputLayerSpaceNetwork->d_bias.Pointer();
			bestWeightsLayers[ll] = selectiveInputLayerSpaceNetwork->d_bestBias.Pointer();
			learnRatesLayers[ll] = selectiveInputLayerSpaceNetwork->d_learnRateBias.Pointer();
			lastDeltaLayers[ll] = selectiveInputLayerSpaceNetwork->d_lastDeltaBias.Pointer();
			lastDeltaWithoutLMlayers[ll] = selectiveInputLayerSpaceNetwork->d_lastDeltaWithoutLearningMomentumBias.Pointer();
			ll++;
		}
	}

	d_numberWeightsLayer = numberWeightsLayer;
	d_weightsLayers = weightsLayers;
	d_bestWeightsLayers = bestWeightsLayers;
	d_learnRatesLayers = learnRatesLayers;
	d_lastDeltaLayers = lastDeltaLayers;
	d_lastDeltaWithoutLMlayers = lastDeltaWithoutLMlayers;

	robustLearning = true;
	rmsGrowToApplyRobustLearning = CUDA_VALUE(1.001); // 0.1%
	robustFactor = CUDA_VALUE(0.5);
	momentum = CUDA_VALUE(0.7);
	u = CUDA_VALUE(1.2);
	d = CUDA_VALUE(0.8);
	maxStepSize = CUDA_VALUE(10.0);

	//Create the RMS vectors	
	int sizeRMSvector = (layers[outputLayer].connections > MAX_THREADS_PER_BLOCK) ? Layer::patterns * layers[outputLayer].neurons : Layer::patterns;
	d_rms.ResizeWithoutPreservingData(sizeRMSvector);

	layers[outputLayer].d_desOutputs = d_desOutputs.Pointer();
	layers[outputLayer].d_rms = d_rms.Pointer();
	layers[outputLayer].sharedMemFire += layers[outputLayer].neurons * sizeof(cudafloat);

	// Initialize the initial RMS
	HostArray<cudafloat> h_bestRMS(1);
	h_bestRMS[0] = CUDA_VALUE(1.0);
	d_bestRMS = h_bestRMS;
	rms.Value() = h_bestRMS[0];

	//Other stuff
	patternsBlockSize = 1;
	while(patternsBlockSize < MAX_THREADS_PER_BLOCK && patternsBlockSize < Layer::patterns) patternsBlockSize <<= 1;

	numberPatternsNeurons = (cudafloat) Layer::patterns * (cudafloat) layers[outputLayer].neurons;

	epoch = 0;
}

BackPropagation::BackPropagation(HostArray<int> & sizeLayers, HostMatrix<cudafloat> & trainInputPatterns, HostMatrix<cudafloat> & trainDesiredOutputPatterns, cudafloat initialLearningRate) {
	CreateNetwork(sizeLayers, nullptr, nullptr, trainInputPatterns, trainDesiredOutputPatterns, initialLearningRate);
}

HostArray<cudafloat> BackPropagation::GetLayerWeights(int layer) {
	assert(layer >= 0 && layer < layers.Length());
	return HostArray<cudafloat>(layers[layer].d_weights);
}

void BackPropagation::SetLayerWeights(int layer, HostArray<cudafloat> & weights) {
	assert(layer >= 0 && layer < layers.Length());
	layers[layer].d_weights = weights;
}

void BackPropagation::SetLayerWeights(int layer, HostMatrix<cudafloat> & weights, HostArray<cudafloat> & bias) {
	assert(layer >= 0 && layer < layers.Length());

	Layer * l = &(layers[layer]);
	int neurons = l->neurons;
	int inputs = weights.Columns();

	assert(neurons == bias.Length());

	HostArray<cudafloat> weights_bias(weights.Elements() + bias.Length());

	int w = 0;
	for(int n = 0; n < neurons; n++) {
		weights_bias[w++] = bias[n];
		for(int i = 0; i < inputs; i++) {
			weights_bias[w++] = weights(n, i);
		}
	}

	layers[layer].d_weights = weights_bias;
}

HostArray<cudafloat> BackPropagation::GetSelectiveInputWeights() {
	return HostArray<cudafloat>(selectiveInputLayer->d_weights);
}

void BackPropagation::SetSelectiveInputWeights(HostArray<cudafloat> & weights) {
	selectiveInputLayer->d_weights = weights;
}

HostArray<cudafloat> BackPropagation::GetSelectiveInputBias() {
	return HostArray<cudafloat>(selectiveInputLayer->d_bias);
}

void BackPropagation::SetSelectiveInputBias(HostArray<cudafloat> & bias) {
	selectiveInputLayer->d_bias = bias;
}

void BackPropagation::RandomizeWeights(cudafloat minValue, cudafloat maxValue) {
	int nSpaceLayers = spaceLayers.Length();
	for (int layer = 0; layer < nSpaceLayers; layer++) spaceLayers[layer].RandomizeWeights(minValue, maxValue, initialLearningRate);

	int nLayers = layers.Length();
	for (int layer = 0; layer < nLayers; layer++) layers[layer].RandomizeWeights(minValue, maxValue, initialLearningRate);

	if (selectiveInputLayerSpaceNetwork) selectiveInputLayerSpaceNetwork->RandomizeWeights(minValue, maxValue, initialLearningRate, selectiveInputs);
	if (selectiveInputLayer) selectiveInputLayer->RandomizeWeights(minValue, maxValue, initialLearningRate, selectiveInputs);
	epoch = 0;
}

bool BackPropagation::GetRobustLearning() const { 
	return robustLearning;
}

void BackPropagation::SetRobustLearning(bool value) { 
	robustLearning = value;
}

cudafloat BackPropagation::GetMaxPercentageRMSGrow() const {
	return rmsGrowToApplyRobustLearning - CUDA_VALUE(1.0);
}

void BackPropagation::SetMaxPercentageRMSGrow(cudafloat value) {
	assert(value > CUDA_VALUE(0.0));
	rmsGrowToApplyRobustLearning = CUDA_VALUE(1.0) + value;
}

cudafloat BackPropagation::GetRobustFactor() const {
	return robustFactor;
}

void BackPropagation::SetRobustFactor(cudafloat value) {
	assert(value > CUDA_VALUE(0.0) && value < CUDA_VALUE(1.0));
	robustFactor = value;
}

cudafloat BackPropagation::GetMomentum() const {
	return momentum;
}

void BackPropagation::SetMomentum(cudafloat value) {
	assert(value > CUDA_VALUE(0.0) && value < CUDA_VALUE(1.0));
	momentum = value;
}

cudafloat BackPropagation::GetUpStepSizeFactor() const {
	return u;
}

void BackPropagation::SetUpStepSizeFactor(cudafloat value){
	assert(value > CUDA_VALUE(1.0));
	u = value;
}

cudafloat BackPropagation::GetDownStepSizeFactor() const {
	return d;
}

void BackPropagation::SetDownStepSizeFactor(cudafloat value) {
	assert(value > CUDA_VALUE(0.0) && value < CUDA_VALUE(1.0));
	d = value;
}

cudafloat BackPropagation::GetMaxStepSize() const {
	return maxStepSize;
}

void BackPropagation::SetMaxStepSize(cudafloat value) {
	assert(value > CUDA_VALUE(0.0));
	maxStepSize = value;
}

int BackPropagation::GetEpoch() const {
	return epoch;
}

int BackPropagation::GetNumberLayers() const {
	return layers.Length();
}

int BackPropagation::GetNumberInputs() const {
	return layers[0].inputsWithoutBias;
}

int BackPropagation::GetNumberOutputs() const {
	return layers[layers.Length() - 1].neurons;
}

int BackPropagation::GetNumberNeurons(int layer) const {
	assert(layer >= 0 && layer < layers.Length());
	return layers[layer].neurons;
}

void BackPropagation::Fire() {
	if (selectiveInputLayerSpaceNetwork != nullptr) selectiveInputLayerSpaceNetwork->Fire(streamKernels);

	int nSpaceLayers = spaceLayers.Length();
	for (int l = 0; l < nSpaceLayers; l++) spaceLayers[l].Fire(streamKernels);

	if (selectiveInputLayer != nullptr) selectiveInputLayer->Fire(streamKernels);

	int numLayers = layers.Length();
	for(int l = 0; l < numLayers; l++) layers[l].Fire(streamKernels);
}

cudafloat BackPropagation::GetRMS() {
	cudaDeviceSynchronize();

	Fire(); // Determine the network outputs

	// Calculate the RMS 
	KernelCalculateRMS(streamKernels, patternsBlockSize, d_rms.Pointer(), d_rmsOut.Pointer(), d_rms.Length(), numberPatternsNeurons);
	rms.UpdateValue(d_rmsOut.Pointer());

	return rms.Value();
}

cudafloat BackPropagation::GetRMSestimate() {
	cudafloat RMS = rms.Value();

	if (epoch == 0 && RMS >= CUDA_VALUE(1.0)) return GetRMS();

	return RMS;
}

void BackPropagation::TrainOneEpoch() {
	int numLayers = layers.Length();
	int nSpaceLayers = spaceLayers.Length();

	Fire(); // Determine the network outputs

	// Calculate the RMS / Robust training
	if (robustLearning) {
		KernelCalculateRMS(streamKernels, patternsBlockSize, d_rms.Pointer(), d_rmsOut.Pointer(), d_rms.Length(), numberPatternsNeurons);
		if (cudaStreamQuery(streamRMS) == cudaSuccess) rms.UpdateValueAsync(d_rmsOut.Pointer(), streamRMS);
		
		RobustLearning<<<1, maxNumberWeigths, 0, streamKernels>>>(d_rmsOut.Pointer(), d_bestRMS.Pointer(), (cudafloat) rmsGrowToApplyRobustLearning, layersRobustTraining, d_numberWeightsLayer.Pointer(), d_weightsLayers.Pointer(), d_bestWeightsLayers.Pointer(), d_learnRatesLayers.Pointer(), robustFactor, d_lastDeltaWithoutLMlayers.Pointer(), d_lastDeltaLayers.Pointer());
	} else {
		if (cudaStreamQuery(streamRMS) == cudaSuccess) {
			KernelCalculateRMS(streamRMS, patternsBlockSize, d_rms.Pointer(), d_rmsOut.Pointer(), d_rms.Length(), numberPatternsNeurons);
			rms.UpdateValueAsync(d_rmsOut.Pointer(), streamRMS);
		}
	}

	// Calculate local gradients. The local gradient for the output layer was already calculated.
	cudafloat * rms = (robustLearning) ? d_rmsOut.Pointer() : nullptr;
	cudafloat * bestRMS = (robustLearning) ? d_bestRMS.Pointer() : nullptr;	

	for(int l = numLayers - 2; l >= 0; l--) {
		layers[l].CalculateLocalGradient(streamKernels, rms, bestRMS, rmsGrowToApplyRobustLearning, layers[l + 1]);
	}

	if (selectiveInputLayer != nullptr) selectiveInputLayer->CalculateLocalGradient(streamKernels, rms, bestRMS, rmsGrowToApplyRobustLearning, layers[0]);

	for (int l = nSpaceLayers -2; l >= 0; l--) spaceLayers[l].CalculateLocalGradient(streamKernels, rms, bestRMS, rmsGrowToApplyRobustLearning, spaceLayers[l + 1]);

	if (selectiveInputLayerSpaceNetwork != nullptr) selectiveInputLayerSpaceNetwork->CalculateLocalGradient(streamKernels, rms, bestRMS, rmsGrowToApplyRobustLearning, spaceLayers[0]);

	// Correct the weights
	for(int l = numLayers - 1; l >= 0; l--) {
		layers[l].CorrectWeights(streamKernels, patternsBlockSize, rms, bestRMS, rmsGrowToApplyRobustLearning, robustFactor, momentum, u, d, maxStepSize);
	}

	if (selectiveInputLayer != nullptr) selectiveInputLayer->CorrectWeights(streamKernels, rms, bestRMS, rmsGrowToApplyRobustLearning, robustFactor, momentum, u, d, maxStepSize);

	for (int l = nSpaceLayers - 1; l >= 0; l--) spaceLayers[l].CorrectWeights(streamKernels, patternsBlockSize, rms, bestRMS, rmsGrowToApplyRobustLearning, robustFactor, momentum, u, d, maxStepSize);

	if (selectiveInputLayerSpaceNetwork != nullptr) selectiveInputLayerSpaceNetwork->CorrectWeights(streamKernels, rms, bestRMS, rmsGrowToApplyRobustLearning, robustFactor, momentum, u, d, maxStepSize);
		
	epoch++;
}

void BackPropagation::Train(int epochs) {
	for (int e = 0; e < epochs; e++) TrainOneEpoch();
}

void BackPropagation::Train(int epochs, cudafloat rmsStop) {
	// In some situations, we may get the RMS error from a previous trained network.
	// To avoid this, we compute the actual RMS before training the network.
	GetRMS();

	for (int e = 0; e < epochs; e++) {		
		TrainOneEpoch();
		if (GetRMSestimate() <= rmsStop) break;
	}
}

HostMatrix<cudafloat> BackPropagation::GetOutputs(HostMatrix<cudafloat> & inputs) {
	int patterns = inputs.Rows();
	int numberLayers = layers.Length();
	int numberSpaceLayers = spaceLayers.Length();

	DeviceMatrix<cudafloat> d_inputs(inputs);

	HostArray< DeviceMatrix<cudafloat> * > spaceLayerOutputs;
	spaceLayerOutputs.ResizeWithoutPreservingData(numberSpaceLayers);
	for (int l = 0; l < numberSpaceLayers; l++) {
		spaceLayerOutputs[l] = new DeviceMatrix<cudafloat>(patterns, spaceLayers[l].neurons);
	}

	HostArray< DeviceMatrix<cudafloat> * > layerOutputs;
	layerOutputs.ResizeWithoutPreservingData(numberLayers);
	for (int l = 0; l < numberLayers; l++) {
		layerOutputs[l] = new DeviceMatrix<cudafloat>(patterns, layers[l].neurons);
	}

	cudafloat * layerInputs = d_inputs.Pointer();

	int ninputs = d_inputs.Columns();
	DeviceArray<cudafloat> outputsSelectiveInput(patterns * ninputs);

	if (selectiveInputLayerSpaceNetwork != nullptr) {
		int processed = 0;
		do {
			int patternsToProcess = (patterns > 65535) ? 65535 : patterns;
			FireSelectiveInputs<<<patternsToProcess, ninputs, 0, streamKernels>>>(layerInputs + (processed * ninputs), selectiveInputLayerSpaceNetwork->d_weights.Pointer(), selectiveInputLayerSpaceNetwork->d_bias.Pointer(), outputsSelectiveInput.Pointer() + (processed * ninputs), ninputs);
			processed += patternsToProcess;
		} while (processed < patterns);

		layerInputs = outputsSelectiveInput.Pointer();
	}

	for (int l = 0; l < numberSpaceLayers; l++) {
		if(spaceLayers[l].connections > MAX_THREADS_PER_BLOCK) {
			dim3 dimNeuronsPatterns;
			dimNeuronsPatterns.x = spaceLayers[l].neurons;

			int processed = 0;
			do {
				int patternsToProcess = (patterns > 65535) ? 65535 : patterns;				
				dimNeuronsPatterns.y = patternsToProcess;
				KernelFireLayer(streamKernels, dimNeuronsPatterns, spaceLayers[l].inputsBlockSize, layerInputs + (processed * spaceLayers[l].inputsWithoutBias), spaceLayers[l].d_weights.Pointer(), nullptr, 0, Layer::totalNeuronsWithSelectiveActivation, spaceLayerOutputs[l]->Pointer() + (processed * spaceLayers[l].inputsWithoutBias), spaceLayers[l].inputsWithoutBias);
				processed += patternsToProcess;
			} while (processed < patterns);
		} else {
			int processed = 0;
			do {
				int patternsToProcess = (patterns > 65535) ? 65535 : patterns;
				FireLayer<<<patternsToProcess, spaceLayers[l].dimInputsNeurons, spaceLayers[l].sharedMemFire, streamKernels>>>(layerInputs + (processed * spaceLayers[l].inputsWithoutBias), spaceLayers[l].d_weights.Pointer(), nullptr, 0, Layer::totalNeuronsWithSelectiveActivation, spaceLayerOutputs[l]->Pointer() + (processed * spaceLayers[l].inputsWithoutBias));
				processed += patternsToProcess;
			} while (processed < patterns);
		}

		layerInputs = spaceLayerOutputs[l]->Pointer();
	}

	cudafloat * d_m = nullptr;
	if (numberSpaceLayers > 0) d_m = layerInputs;

	layerInputs = d_inputs.Pointer();

	if (selectiveInputLayer != nullptr) {
		int processed = 0;
		do {
			int patternsToProcess = (patterns > 65535) ? 65535 : patterns;
			FireSelectiveInputs<<<patternsToProcess, ninputs, 0, streamKernels>>>(layerInputs + (processed * ninputs), selectiveInputLayer->d_weights.Pointer(), selectiveInputLayer->d_bias.Pointer(), outputsSelectiveInput.Pointer() + (processed * ninputs), ninputs);
			processed += patternsToProcess;
		} while (processed < patterns);

		layerInputs = outputsSelectiveInput.Pointer();
	}
		
	for (int l = 0; l < numberLayers; l++) {
		if(layers[l].connections > MAX_THREADS_PER_BLOCK) {
			dim3 dimNeuronsPatterns;
			dimNeuronsPatterns.x = layers[l].neurons;

			int processed = 0;
			do {
				int patternsToProcess = (patterns > 65535) ? 65535 : patterns;				
				dimNeuronsPatterns.y = patternsToProcess;
				KernelFireLayer(streamKernels, dimNeuronsPatterns, layers[l].inputsBlockSize, layerInputs + (processed * layers[l].inputsWithoutBias), layers[l].d_weights.Pointer(), (layers[l].d_m != nullptr) ? d_m + (processed * Layer::totalNeuronsWithSelectiveActivation) : nullptr, layers[l].mOffset, Layer::totalNeuronsWithSelectiveActivation, layerOutputs[l]->Pointer() + (processed * layers[l].inputsWithoutBias), layers[l].inputsWithoutBias);
				processed += patternsToProcess;
			} while (processed < patterns);
		} else {
			int processed = 0;
			do {
				int patternsToProcess = (patterns > 65535) ? 65535 : patterns;
				FireLayer<<<patternsToProcess, layers[l].dimInputsNeurons, layers[l].sharedMemFire, streamKernels>>>(layerInputs + (processed * layers[l].inputsWithoutBias), layers[l].d_weights.Pointer(), (layers[l].d_m != nullptr) ? d_m + (processed * Layer::totalNeuronsWithSelectiveActivation) : nullptr, layers[l].mOffset, Layer::totalNeuronsWithSelectiveActivation, layerOutputs[l]->Pointer() + (processed * layers[l].inputsWithoutBias));
				processed += patternsToProcess;
			} while (processed < patterns);
		}

		layerInputs = layerOutputs[l]->Pointer();
	}

	HostMatrix<cudafloat> outputs(*(layerOutputs[numberLayers - 1]));

	for (int l = 0; l < numberSpaceLayers; l++) {
		delete spaceLayerOutputs[l];
	}

	for (int l = 0; l < numberLayers; l++) {
		delete layerOutputs[l];
	}

	return outputs;
}

}