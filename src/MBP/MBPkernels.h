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

#ifndef GPUMLib_MBPkernels_h
#define GPUMLib_MBPkernels_h

#include <cuda_runtime.h>
#include "../common/config.h"
#include "../common/CudaDefinitions.h"

namespace GPUMLib {

//! \addtogroup mbpkernels Back-Propagation and Multiple Back-Propagation kernels
//! @{

//! Kernel to calculate the outputs of all neurons in a given layer. 
//! For the output layer use FireOutputLayer or KernelFireOutputLayer instead.
//! \attention used only if the number of connections (weights) is NOT greather than MAX_THREADS_PER_BLOCK.
//! \param[in] inputs Input data for all patterns (numInputs * numPatterns). Position [p * numInputs + i]  should contain the input i for pattern p.
//! \param[in] weights Input weights of the layer (numNeurons * (numInputs + 1)). Position [n * (numInputs + 1)] should contain the bias of neuron n. Position [n * (numInputs + 1) + i + 1] should contain the weight of the connection between input (neuron) i and neuron n.
//! \param[in] m Importance of the neurons of the layer (numNeurons * numPatterns). Should be nullptr for neurons without selective activation. Position [p * numNeurons + n] should contain the importance of neuron n for pattern p.
//! \param[in] mOffset Offset of the importance of this layers neurons whithin the importance values. Tipically identical to the number of neurons in the previous layers.
//! \param[in] totalNeuronsWithSelectiveActivation Number of neurons with selective activation in all layers (total).
//! \param[out] outputs Neuron outputs of the layer (numNeurons * numPatterns). Position [p * numNeurons + n] will contain the output of neuron n for pattern p.
//! \par Examples:
//! \code 
//! FireLayer<<<numPatterns, (numInputs + 1, numNeurons), numWeights * sizeof(cudafloat), stream>>>(inputs, weights, m, outputs);
//! \endcode
//! \code 
//! FireLayer<<<numPatterns, (numInputs + 1, numNeurons), numWeights * sizeof(cudafloat), stream>>>(inputs, weights, nullptr, outputs);
//! \endcode
//! \sa KernelFireLayer
//! \sa FireOutputLayer
//! \sa KernelFireOutputLayer
//! \sa MAX_THREADS_PER_BLOCK
KERNEL FireLayer(cudafloat * inputs, cudafloat * weights, cudafloat * m, int mOffset, int totalNeuronsWithSelectiveActivation, cudafloat * outputs);

//! Kernel to calculate the outputs of all neurons in a given layer. 
//! For the output layer use FireOutputLayer or KernelFireOutputLayer instead.
//! \attention Used when the number of connections (weights) is greather than MAX_THREADS_PER_BLOCK. Use FireLayer otherwise.
//! \param[in] stream CUDA stream.
//! \param[in] gridSize Size of the grid (numNeurons, numPatterns).
//! \param[in] blockSize Block size. Must be a multiple of 2 and not exceed MAX_THREADS_PER_BLOCK.
//! \param[in] inputs Input data for all patterns (numInputs * numPatterns). Position [p * numInputs + i]  should contain the input i for pattern p.
//! \param[in] weights Input weights of the layer (numNeurons * (numInputs + 1)). Position [n * (numInputs + 1)] should contain the bias of neuron n. Position [n * (numInputs + 1) + i + 1] should contain the weight of the connection between input (neuron) i and neuron n.
//! \param[in] m Importance of the neurons of the layer (numNeurons * numPatterns). Should be nullptr for neurons without selective activation. Position [p * numNeurons + n] should contain the importance of neuron n for pattern p.
//! \param[in] mOffset Offset of the importance of this layers neurons whithin the importance values. Tipically identical to the number of neurons in the previous layers.
//! \param[in] totalNeuronsWithSelectiveActivation Number of neurons with selective activation in all layers (total).
//! \param[out] outputs Neuron outputs of the layer (numNeurons * numPatterns). Position [p * numNeurons + n] will contain the output of neuron n for pattern p.
//! \param[in] numInputs Number of inputs.
//! \par Examples:
//! \code 
//! dim3 dimNeuronsPatterns(numNeurons, numPatterns);
//! KernelFireLayer(stream, dimNeuronsPatterns, 512, inputs, weights, m, outputs, numInputs);
//! \endcode
//! \code 
//! dim3 dimNeuronsPatterns(numNeurons, numPatterns);
//! KernelFireLayer(stream, dimNeuronsPatterns, 512, inputs, weights, nullptr, outputs, numInputs);
//! \endcode
//! \sa FireLayer
//! \sa FireOutputLayer
//! \sa KernelFireOutputLayer
//! \sa MAX_THREADS_PER_BLOCK
void KernelFireLayer(cudaStream_t stream, dim3 & gridSize, int blockSize, cudafloat * inputs, cudafloat * weights, cudafloat * m, int mOffset, int totalNeuronsWithSelectiveActivation, cudafloat * outputs, int numInputs);

//! Kernel to calculate the outputs of the network output layer and the local gradients of its neurons. 
//! If the layer contains selective activation neurons, the local gradients of the corresponding space network neurons are also calculated.
//! This kernel also calculates part of the RMS.
//! \attention used only if the number of connections (weights) is NOT greather than MAX_THREADS_PER_BLOCK.
//! \param[in] inputs Input data for all patterns (numInputs * numPatterns). Position [p * numInputs + i]  should contain the input i for pattern p.
//! \param[in] weights Input weights of the layer (numNeurons * (numInputs + 1)). Position [n * (numInputs + 1)] should contain the bias of neuron n. Position [n * (numInputs + 1) + i + 1] should contain the weight of the connection between input (neuron) i and neuron n.
//! \param[in] m Importance of the neurons of the layer (numNeurons * numPatterns). Should be nullptr for neurons without selective activation. Position [p * numNeurons + n] should contain the importance of neuron n for pattern p.
//! \param[in] mOffset Offset of the importance of this layers neurons whithin the importance values. Tipically identical to the number of neurons in the previous layers.
//! \param[in] totalNeuronsWithSelectiveActivation Number of neurons with selective activation in all layers (total).
//! \param[in] desiredOutputs The network desired outputs for all patterns (numNeurons * numPatterns). Position [p * numNeurons + n] should contain the desired output of neuron n for pattern p.
//! \param[out] outputs Network outputs (numNeurons * numPatterns). Position [p * numNeurons + n] will contain the output of neuron n for pattern p.
//! \param[out] localGradient Local gradients of the network output layer (numNeurons * numPatterns). Position [p * numNeurons + n] will contain the local gradients of neuron n for pattern p.
//! \param[out] rms partial calculation of the root mean square error (numPatterns).
//! \param[out] localGradientSpaceNet Local gradients of the associated neurons in space network output layer (numNeurons * numPatterns). Should be nullptr for neurons without selective activation. Position [p * numNeurons + n] will contain the local gradients, for pattern p, of the neuron that calculates the importance of neuron n.
//! \par Examples:
//! \code 
//! FireOutputLayer<<<NumPatterns, (numInputs + 1, numNeurons), (numWeights + numNeurons) * sizeof(cudafloat), stream>>>(inputs, weights, m, desiredOutputs, outputs, localGradient, rms, localGradientSpaceNet);
//! \endcode
//! \code 
//! FireOutputLayer<<<NumPatterns, (numInputs + 1, numNeurons), (numWeights + numNeurons) * sizeof(cudafloat), stream>>>(inputs, weights, nullptr, desiredOutputs, outputs, localGradient, rms, nullptr);
//! \endcode
//! \sa KernelFireOutputLayer
//! \sa MAX_THREADS_PER_BLOCK
KERNEL FireOutputLayer(cudafloat * inputs, cudafloat * weights, cudafloat * m, int mOffset, int totalNeuronsWithSelectiveActivation, cudafloat * desiredOutputs, cudafloat * outputs, cudafloat * localGradient, cudafloat * rms, cudafloat * localGradientSpaceNet);

//! Kernel to calculate the outputs of the network output layer and the local gradients of its neurons. 
//! If the layer contains selective activation neurons, the local gradients of the corresponding space network neurons are also calculated.
//! This kernel also calculates part of the RMS.
//! \attention Used when the number of connections (weights) is greather than MAX_THREADS_PER_BLOCK. Use FireOutputLayer otherwise.
//! \param[in] stream CUDA stream.
//! \param[in] gridSize Size of the grid (numNeurons, numPatterns).
//! \param[in] blockSize Block size. Must be a multiple of 2 and not exceed MAX_THREADS_PER_BLOCK.
//! \param[in] inputs Input data for all patterns (numInputs * numPatterns). Position [p * numInputs + i]  should contain the input i for pattern p.
//! \param[in] weights Input weights of the layer (numNeurons * (numInputs + 1)). Position [n * (numInputs + 1)] should contain the bias of neuron n. Position [n * (numInputs + 1) + i + 1] should contain the weight of the connection between input (neuron) i and neuron n.
//! \param[in] m Importance of the neurons of the layer (numNeurons * numPatterns). Should be nullptr for neurons without selective activation. Position [p * numNeurons + n] should contain the importance of neuron n for pattern p.
//! \param[in] mOffset Offset of the importance of this layers neurons whithin the importance values. Tipically identical to the number of neurons in the previous layers.
//! \param[in] totalNeuronsWithSelectiveActivation Number of neurons with selective activation in all layers (total).
//! \param[in] desiredOutputs The network desired outputs for all patterns (numNeurons * numPatterns). Position [p * numNeurons + n] should contain the desired output of neuron n for pattern p.
//! \param[out] outputs Network outputs (numNeurons * numPatterns). Position [p * numNeurons + n] will contain the output of neuron n for pattern p.
//! \param[out] localGradient Local gradients of the network output layer (numNeurons * numPatterns). Position [p * numNeurons + n] will contain the local gradients of neuron n for pattern p.
//! \param[out] rms partial calculation of the root mean square error (numPatterns).
//! \param[out] localGradientSpaceNet Local gradients of the associated neurons in space network output layer (numNeurons * numPatterns). Should be nullptr for neurons without selective activation. Position [p * numNeurons + n] will contain the local gradients, for pattern p, of the neuron that calculates the importance of neuron n.
//! \param[in] numInputs Number of inputs.
//! \par Examples:
//! \code 
//! dim3 dimNeuronsPatterns(numNeurons, numPatterns);
//! KernelFireOutputLayer(stream, dimNeuronsPatterns, 512, inputs, weights, m, desiredOutputs, outputs, localGradient, rms, localGradientSpaceNet, numInputs);
//! \endcode
//! \code 
//! dim3 dimNeuronsPatterns(numNeurons, numPatterns);
//! KernelFireOutputLayer(stream, dimNeuronsPatterns, 512, inputs, weights, nullptr, desiredOutputs, outputs, localGradient, rms, nullptr, numInputs);
//! \endcode
//! \sa FireOutputLayer
//! \sa MAX_THREADS_PER_BLOCK
void KernelFireOutputLayer(cudaStream_t stream, dim3 & gridSize, int blockSize, cudafloat * inputs, cudafloat * weights, cudafloat * m, int mOffset, int totalNeuronsWithSelectiveActivation, cudafloat * desiredOutputs, cudafloat * outputs, cudafloat * localGradient, cudafloat * rms, cudafloat * localGradientSpaceNet, int numInputs);

//! Kernel to calculate the local gradient of all neurons in a hidden layer.
//! For selective activation neurons, the local gradients of the corresponding space network neurons are also calculated.
//! \attention used only for hidden layers. The local gradients of output layers are automatically calculated by FireOutputLayer or by KernelFireOutputLayer.
//! \param[in] rmsF address of the variable containg the current root mean square error. Should be nullptr if robust training is not used.
//! \param[in] bestRMS address of the variable containg the best root mean square error found so far. Should be nullptr if robust training is not used.
//! \param[in] maxErrorGrowth How much the root mean square error is allowed to grow relatively to bestRMS before applying the robusteness techique (see the RobustLearning kernel).
//! \param[in] outputs Network outputs (numNeurons * numPatterns). Position [p * numNeurons + n] contains the output of neuron n for pattern p.
//! \param[in] weights Input weights of the layer (numNeurons * (numInputs + 1)). Position [n * (numInputs + 1)] should contain the bias of neuron n. Position [n * (numInputs + 1) + i + 1] should contain the weight of the connection between input (neuron) i and neuron n.
//! \param[in] m Importance of the neurons of the layer (numNeurons * numPatterns). Should be nullptr for neurons without selective activation. Position [p * numNeurons + n] should contain the importance of neuron n for pattern p.
//! \param[in] mOffset Offset of the importance of this layers neurons whithin the importance values. Tipically identical to the number of neurons in the previous layers.
//! \param[in] totalNeuronsWithSelectiveActivation Number of neurons with selective activation in all layers (total).
//! \param[in] localGradientNextLayer Local gradients of the next layer (numOutputs * numPatterns). Position [p * numOutputs + o] will contain the local gradients of neuron o for pattern p.
//! \param[out] localGradient Local gradients of the layer (numNeurons * numPatterns). Position [p * numNeurons + n] will contain the local gradients of neuron n for pattern p.
//! \param[out] localGradientSpaceNet Local gradients of the associated neurons in space network output layer (numNeurons * numPatterns). Should be nullptr for neurons without selective activation. Position [p * numNeurons + n] will contain the local gradients, for pattern p, of the neuron that calculates the importance of neuron n.
//! \par Examples:
//! \code 
//! dim3 dimOutputsNeurons(numOutputs, numNeurons);
//! CalculateLocalGradient<<<patterns, dimOutputsNeurons, (outputs * (neurons + 1)) * sizeof(cudafloat)>>>(rms, bestRMS, maxErrorGrowthToApplyRobustLearning, outputs, nextLayerWeights, m, nextLayerlocalGradient, localGradient, lgSpaceNet);
//! \endcode
//! \code 
//! dim3 dimOutputsNeurons(numOutputs, numNeurons);
//! CalculateLocalGradient<<<patterns, dimOutputsNeurons, (outputs * (neurons + 1)) * sizeof(cudafloat)>>>(rms, bestRMS, maxErrorGrowthToApplyRobustLearning, outputs, nextLayerWeights, nullptr, nextLayerlocalGradient, localGradient, nullptr);
//! \endcode
//! \sa FireOutputLayer
//! \sa KernelFireOutputLayer
//! \sa RobustLearning
KERNEL CalculateLocalGradient(cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, cudafloat * outputs, cudafloat * weights, cudafloat * m, int mOffset, int totalNeuronsWithSelectiveActivation, cudafloat * localGradientNextLayer, cudafloat * localGradient, cudafloat * localGradientSpaceNet);

//! Kernel to adjust the weights of a given layer. The step sizes are also updated.
//! \param[in] stream CUDA stream.
//! \param[in] gridSize Size of the grid (numInputs + 1, numNeurons).
//! \param[in] blockSize Block size. Must be a multiple of 2 and not exceed MAX_THREADS_PER_BLOCK.
//! \param[in] rmsF address of the variable containg the current root mean square error. Should be nullptr if robust training is not used.
//! \param[in] bestRMS address of the variable containg the best root mean square error found so far. Should be nullptr if robust training is not used.
//! \param[in] maxErrorGrowth How much the root mean square error is allowed to grow relatively to bestRMS before applying the robusteness techique (see the RobustLearning kernel).
//! \param[in] inputs Input data for all patterns (numInputs * numPatterns). Position [p * numInputs + i]  should contain the input i for pattern p.
//! \param[in] localGradient Local gradients of the layer (numNeurons * numPatterns). Position [p * numNeurons + n] should contain the local gradients of neuron n for pattern p.
//! \param[in, out] weights Input weights of the layer (numNeurons * (numInputs + 1)). Position [n * (numInputs + 1)] should contain the bias of neuron n. Position [n * (numInputs + 1) + i + 1] should contain the weight of the connection between input (neuron) i and neuron n.
//! \param[in, out] learningRate Learning rate (step size) of each input connections of the layer (numNeurons * (numInputs + 1)). Position [n * (numInputs + 1)] should contain the step size of the bias of neuron n. Position [n * (numInputs + 1) + i + 1] should contain the step size of the connection between input (neuron) i and neuron n.
//! \param[in, out] lastDeltaWithoutLearningMomentum Last delta without learning or momentum of each input connections of the layer (numNeurons * (numInputs + 1)). Position [n * (numInputs + 1)] should contain the last delta associated with the bias of neuron n. Position [n * (numInputs + 1) + i + 1] should contain the last delta associated with the connection between input (neuron) i and neuron n.
//! \param[in, out] lastDelta Last delta of each input connections of the layer (numNeurons * (numInputs + 1)). Position [n * (numInputs + 1)] should contain the last delta associated with the bias of neuron n. Position [n * (numInputs + 1) + i + 1] should contain the last delta associated with the connection between input (neuron) i and neuron n.
//! \param[in] maxStepSize Maximum step size.
//! \param[in] u Increment factor (step size).
//! \param[in] d Decrement factor (step size).
//! \param[in] r Reducing factor (step size / robust training).
//! \param[in] momentum Momentum factor.
//! \param[in] numberPatterns Number of patterns.
//! \par Examples:
//! \code 
//! dim3 dimInputsNeurons(numInputs + 1, numNeurons);
//! KernelCorrectLayerWeights(stream, dimInputsNeurons, patternsBlockSize, rms, bestRMS, maxErrorGrowth, inputValues, localGradient, weights, learnRate, lastDeltaWithoutLearningMomentum, lastDelta, maxStepSize, u, d, robustFactor, momentum, numPatterns);
//! \endcode
void KernelCorrectLayerWeights(cudaStream_t stream, dim3 & gridSize, int blockSize, cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, cudafloat * inputs, cudafloat * localGradient, cudafloat * weights, cudafloat * learningRate, cudafloat * lastDeltaWithoutLearningMomentum, cudafloat * lastDelta, cudafloat maxStepSize, cudafloat u, cudafloat d, cudafloat r, cudafloat momentum, int numberPatterns);

//! Kernel to calculate the Root Mean Square (RMS) error of the neural network.
//! \param[in] stream CUDA stream.
//! \param[in] blockSize Block size. Must be a multiple of 2 and not exceed MAX_THREADS_PER_BLOCK.
//! \param[in] rms The partial calculations of the root mean square error (see FireOutputLayer and KernelFireOutputLayer).
//! \param[out] rmsOut Address of the variable that will contain the root mean square error. 
//! \param[in] numberPatterns Number of patterns.
//! \param[in] numberPatternsNeurons Product between the number of patterns and the number of neurons.
//! \par Examples:
//! \code 
//! KernelCalculateRMS(stream, patternsBlockSize, rms, rmsOut, numberPatterns, (cudafloat) numberPatterns * numberNeurons);
//! \endcode
//! \sa KernelFireOutputLayer
//! \sa FireOutputLayer
void KernelCalculateRMS(cudaStream_t stream, int blockSize, cudafloat * rms, cudafloat * rmsOut, int numberPatterns, cudafloat numberPatternsNeurons);

//! This kernel Checks if the RMS is lower than the minimum obtained so far. 
//! If so, the minimum RMS is updated and the NN weights are stored. 
//! Otherwise, the kernel checks whether the RMS exceeded the best RMS by a given tolerance and in affirmative case: 
//! - the best weights are restored;
//! - the step sizes reduced by a given factor;
//! - the momentum memories set to zero.
//! \param[in] rmsF address of the variable containg the current root mean square error. Should be nullptr if robust training is not used.
//! \param[in] bestRMS address of the variable containg the best root mean square error found so far. Should be nullptr if robust training is not used.
//! \param[in] maxErrorGrowth How much the root mean square error is allowed to grow relatively to bestRMS before applying the robusteness techique (see the RobustLearning kernel).
//! \param[in] layers Number of layers of the network (not including the input layers). Must include the layers of both the main and the space network.
//! \param[in] numberWeights Number of input weights of each layer.
//! \param[in] weights Weights of each layer.
//! \param[in,out] bestWeights Best weights of each layer.
//! \param[in,out] learningRate Learning rate (step sizes) of the input connections of each layer.
//! \param[in] r Reducing factor (step size).
//! \param[in,out] lastDeltaWithoutLearningMomentum Last delta without learning or momentum of the input connections of each layer.
//! \param[in,out] lastDelta Last delta of the input connections of each layer.
//! \par Examples:
//! \code 
//! RobustLearning<<<1, maxNumberWeightsPerLayer, 0>>>(rms, bestRMS, maxErrorGrowth, numLayers, numberWeightsPerLayer, weightsLayers, bestWeightsLayers, learnRatesLayers, robustFactor,  lastDeltaWithoutLMlayers, lastDeltaLayers);
//! \endcode
KERNEL RobustLearning(cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, int layers, int * numberWeights, cudafloat ** weights, cudafloat ** bestWeights, cudafloat ** learningRate, cudafloat r, cudafloat ** lastDeltaWithoutLearningMomentum, cudafloat ** lastDelta);

//! Kernel to calculate the outputs of the selective input neurons. 
//! \param[in] inputs Input data for all patterns (numInputs * numPatterns). Position [p * numInputs + i]  should contain the input i for pattern p.
//! \param[in] weights Input weights of selective input neurons (one per neuron).
//! \param[in] bias Bias of selective input neurons.
//! \param[out] outputs Neuron outputs of the layer (numNeurons * numPatterns). Position [p * numNeurons + n] will contain the output of neuron n for pattern p.
//! \param[in] numNeurons Number of neurons (identical to the number of inputs of the network).
KERNEL FireSelectiveInputs(cudafloat * inputs, cudafloat * weights, cudafloat * bias, cudafloat * outputs, int numNeurons);

//! Kernel to calculate the local gradient of the selective input neurons.
//! \param[in] rmsF address of the variable containg the current root mean square error. Should be nullptr if robust training is not used.
//! \param[in] bestRMS address of the variable containg the best root mean square error found so far. Should be nullptr if robust training is not used.
//! \param[in] maxErrorGrowth How much the root mean square error is allowed to grow relatively to bestRMS before applying the robusteness techique (see the RobustLearning kernel).
//! \param[in] inputs Input data for all patterns (numInputs * numPatterns). Position [p * numInputs + i]  should contain the input i for pattern p.
//! \param[in] selectiveNeuronsWeights Input weights of selective input neurons (one per neuron).
//! \param[in] selectiveNeuronsBias Bias of selective input neurons.
//! \param[in] weights Input weights of the next layer (numNeurons * (numInputs + 1)). Position [n * (numInputs + 1)] should contain the bias of neuron n. Position [n * (numInputs + 1) + i + 1] should contain the weight of the connection between input (neuron) i and neuron n.
//! \param[in] localGradientNextLayer Local gradients of the next layer (numOutputs * numPatterns). Position [p * numOutputs + o] will contain the local gradients of neuron o for pattern p.
//! \param[out] localGradient Local gradients of the layer (numNeurons * numPatterns). Position [p * numNeurons + n] will contain the local gradients of neuron n for pattern p.
//! \sa RobustLearning
KERNEL CalcLocalGradSelectiveInputs(cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, cudafloat * inputs, cudafloat * selectiveNeuronsWeights, cudafloat * selectiveNeuronsBias, cudafloat * weights, cudafloat * localGradientNextLayer, cudafloat * localGradient);

//! Kernel to adjust the weights of a given layer. The step sizes are also updated.
//! \param[in] stream CUDA stream.
//! \param[in] neurons Number of selective input neurons.
//! \param[in] patterns Number of samples.
//! \param[in] rmsF address of the variable containg the current root mean square error. Should be nullptr if robust training is not used.
//! \param[in] bestRMS address of the variable containg the best root mean square error found so far. Should be nullptr if robust training is not used.
//! \param[in] maxErrorGrowth How much the root mean square error is allowed to grow relatively to bestRMS before applying the robusteness techique (see the RobustLearning kernel).
//! \param[in] inputs Input data for all patterns (numInputs * numPatterns). Position [p * numInputs + i]  should contain the input i for pattern p.
//! \param[in] localGradient Local gradients of the layer (numNeurons * numPatterns). Position [p * numNeurons + n] should contain the local gradients of neuron n for pattern p.
//! \param[in, out] selectiveNeuronsWeights Input weights of selective input neurons (one per neuron).
//! \param[in, out] selectiveNeuronsBias Bias of selective input neurons.
//! \param[in, out] learningRateWeights Learning rate (step size) of the input connection of each neuron. 
//! \param[in, out] learningRateBias Learning rate (step size) of the bias of each neuron.
//! \param[in, out] lastDeltaWithoutLearningMomentumWeights Last delta without learning or momentum of the input connections of each neuron.
//! \param[in, out] lastDeltaWithoutLearningMomentumBias Last delta without learning or momentum of the bias of each neuron.
//! \param[in, out] lastDeltaWeights Last delta of the input connections of each neuron.
//! \param[in, out] lastDeltaBias Last delta of the bias of each neuron.
//! \param[in] u Increment factor (step size).
//! \param[in] d Decrement factor (step size).
//! \param[in] r Reducing factor (step size / robust training).
//! \param[in] maxStepSize Maximum step size.
//! \param[in] momentum Momentum factor.
//! \param[in] numberPatterns Number of samples (patterns).
void KernelCorrectWeightsSelectiveInputs(cudaStream_t stream, int neurons, int patterns, cudafloat * rmsF, cudafloat * bestRMS, cudafloat maxErrorGrowth, cudafloat * inputs, cudafloat * localGradient, cudafloat * selectiveNeuronsWeights, cudafloat * selectiveNeuronsBias, cudafloat * learningRateWeights, cudafloat * learningRateBias, cudafloat * lastDeltaWithoutLearningMomentumWeights, cudafloat * lastDeltaWithoutLearningMomentumBias, cudafloat * lastDeltaWeights, cudafloat * lastDeltaBias, cudafloat u, cudafloat d, cudafloat r, cudafloat maxStepSize, cudafloat momentum, int numberPatterns);

//! @}

}

#endif
