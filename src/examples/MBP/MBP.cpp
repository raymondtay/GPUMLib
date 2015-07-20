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

#define _USE_MATH_DEFINES
#include <math.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <time.h>

using namespace std;

#include "../common/CudaInit.h"
#include "../common/OutputFile.h"
#include "../../MBP/MultipleBackPropagation.h"

#define SPIRAL_DENSITY (1)
#define SPIRAL_POINTS (2 * (96 * SPIRAL_DENSITY + 1))

#define MAX_RADIANS 6.5

#define PRECISION 15

#define INITIAL_LEARNING_RATE (CUDA_VALUE(0.7))
#define RMS_STOP (CUDA_VALUE(0.01))

using namespace GPUMLib;

// rescale the inputs between -1 and 1
cudafloat Rescale(cudafloat value, cudafloat min, cudafloat max) {
	cudafloat valueRescaled = CUDA_VALUE(1.0);

	if (min != max) valueRescaled = CUDA_VALUE(-1.0) + CUDA_VALUE(2.0) * (value - min) / (max - min);

	return valueRescaled;
}

void GenerateTwoSpiralsDataset(HostMatrix<cudafloat> & inputs, HostMatrix<cudafloat> & outputs) {
	int pattern = 0;
	
	double minx;
	double maxx;
	
	double miny;
	double maxy;
	
	for (int i = 0 ; i < SPIRAL_POINTS / 2; i++)  {
		double angle = (i * M_PI) / (16.0 * SPIRAL_DENSITY);
		double radians = MAX_RADIANS * ((104 * SPIRAL_DENSITY) - i) / (104 * SPIRAL_DENSITY);

		double x = radians * cos(angle);
		double y = radians * sin(angle);

		inputs(pattern, 0) = (cudafloat) x;
		inputs(pattern, 1) = (cudafloat) y;        
		outputs(pattern++, 0) = CUDA_VALUE(1.0);

		inputs(pattern, 0) = (cudafloat) -x;
		inputs(pattern, 1) = (cudafloat) -y;
		outputs(pattern++, 0) = CUDA_VALUE(0.0);

		if (i == 0) {
			minx = x;
			maxx = x;

			miny = y;
			maxy = y;
		} else {
			if (x < minx) minx = x; else if (x > maxx) maxx = x;
			if (y < miny) miny = y; else if (y > maxy) maxy = y;
		}
		
		if (-x < minx) minx = -x; else if (-x > maxx) maxx = -x;
		if (-y < miny) miny = -y; else if (-y > maxy) maxy = -y;
	}

	// rescale the inputs between -1 and 1 (the outputs don't need to be rescaled)
	for(int p = 0; p < pattern; p++) {
		inputs(p, 0) = Rescale(inputs(p, 0), (cudafloat) minx, (cudafloat) maxx);
		inputs(p, 1) = Rescale(inputs(p, 1), (cudafloat) miny, (cudafloat) maxy);
	}
}

void SaveNetwork(const char * filename, MultipleBackPropagation & mbp) {
	OutputFile f(filename);
	string s;

	f.WriteLine("Multiple Back-Propagation Version 2.1.4");
	f.WriteLine("To open this file use Multiple Back-Propagation software, downloadable at http://dit.ipg.pt/MBP");
	f.WriteLine("twospirals.txt");
	f.WriteLine(""); // test file

	f.WriteLine("32"); // priority normal
	f.WriteLine("1"); // update screen
		
	f.WriteLine("1"); // delta bar delta
	f.WriteLine(mbp.GetUpStepSizeFactor());
	f.WriteLine(mbp.GetDownStepSizeFactor());
	f.WriteLine(mbp.GetMaxStepSize());

	f.WriteLine(mbp.GetRobustLearning()); 
	f.WriteLine(mbp.GetRobustFactor());
	f.WriteLine(1.0 + mbp.GetMaxPercentageRMSGrow()); // rmsGrowToApplyRobustLearning

	f.WriteLine("0.0"); // weightDecay

	f.WriteLine("0"); // autoUpdateLearning
	f.WriteLine("0"); // autoUpdateMomentum

	f.WriteLine("0.01"); //percentIncDecLearnRate
	f.WriteLine("0.01"); //percentIncDecMomentum
	f.WriteLine("0.01"); //percentIncDecSpaceLearnRate
	f.WriteLine("0.01"); //percentIncDecSpaceMomentum

	f.WriteLine(INITIAL_LEARNING_RATE); //mainNetLearningMomentumInformation.learningRate.value
	f.WriteLine("1000"); //mainNetLearningMomentumInformation.learningRate.decayEpochs
	f.WriteLine("1"); //mainNetLearningMomentumInformation.learningRate.decayPercentage 

	f.WriteLine(mbp.GetMomentum()); //mainNetLearningMomentumInformation.momentum.value
	f.WriteLine("0"); //mainNetLearningMomentumInformation.momentum.decayEpochs
	f.WriteLine("0"); //mainNetLearningMomentumInformation.momentum.decayPercentage

	f.WriteLine(INITIAL_LEARNING_RATE); //spaceNetLearningMomentumInformation.learningRate.value
	f.WriteLine("1000"); //spaceNetLearningMomentumInformation.learningRate.decayEpochs
	f.WriteLine("1"); //spaceNetLearningMomentumInformation.learningRate.decayPercentage

	f.WriteLine(mbp.GetMomentum()); //spaceNetLearningMomentumInformation.momentum.value
	f.WriteLine("0"); //spaceNetLearningMomentumInformation.momentum.decayEpochs
	f.WriteLine("0"); //spaceNetLearningMomentumInformation.momentum.decayPercentage

	f.WriteLine("0"); //epochsStop
	f.WriteLine(RMS_STOP);
	f.WriteLine("1000000"); // numberEpochsToStop

	f.WriteLine("0.0"); //spaceRmsStop

	f.WriteLine("1"); //batchTraining
	f.WriteLine("0"); //randomizePatterns
	
	int numLayers = mbp.GetNumberLayers();


	int networkType = 1; //MBPH
	for(int l = 0; l < numLayers - 1; l++) {
		if (!mbp.HasSelectiveNeurons(l)) {
			networkType = 3; // MBPHO
			break;
		}
	}
	if (networkType == 1 && mbp.HasSelectiveNeurons(numLayers - 1)) networkType = 2; // MBP

	f.WriteLine(networkType);

	//main network
	f.Write(mbp.GetNumberInputs());	
	for(int l = 0; l < numLayers; l++) {
		f.Write("-");
		f.Write(mbp.GetNumberNeurons(l));
	}
	f.WriteLine();

	//space network	additional layers
	int numSpaceLayers = mbp.GetNumberLayersSpaceNetwork();
	if (numSpaceLayers > 1) {
		int lastSpaceLayer = numSpaceLayers - 1;
		for(int l = 0; ; l++) {
			f.Write(mbp.GetNumberNeuronsSpaceNetwork(l));
			if (l == lastSpaceLayer - 1) break;
			f.Write("-");
		}
	}
	f.WriteLine();

	// layers information
	for(int l = 0; l < numLayers; l++) {
		int numNeurons = mbp.GetNumberNeurons(l);

		f.WriteLine(mbp.HasSelectiveNeurons(l) ? numNeurons : 0);

		for(int n = 0; n < numNeurons; n++) {
			f.WriteLine("0"); // ActivationFunction
			f.WriteLine("1.0"); //ActivationFunctionParameter
		}
	}

	// space layers information
	for(int l = 0; l < numSpaceLayers; l++) {
		int numNeurons = mbp.GetNumberNeuronsSpaceNetwork(l);
		for(int n = 0; n < numNeurons; n++) {
			f.WriteLine("0"); // ActivationFunction
			f.WriteLine("1.0"); //ActivationFunctionParameter
		}
	}

	f.WriteLine("0"); //ConnectInputLayerWithOutputLayer main
	f.WriteLine("0"); //ConnectInputLayerWithOutputLayer space

	for(int l = 0; l < numLayers; l++) {
		HostArray<cudafloat> weights = mbp.GetLayerWeights(l);

		int numWeights = weights.Length();
		for(int w = 0; w < numWeights; w++) {
			f.WriteLine(weights[w]);
			f.WriteLine("0.0"); //delta
			f.WriteLine("0.0"); //deltaWithoutLearningMomentum
			f.WriteLine(INITIAL_LEARNING_RATE);
		}	
	}

	for(int l = 0; l < numSpaceLayers; l++) {
		HostArray<cudafloat> weights = mbp.GetLayerWeightsSpaceNetwork(l);

		int numWeights = weights.Length();
		for(int w = 0; w < numWeights; w++) {
			f.WriteLine(weights[w]);
			f.WriteLine("0.0"); //delta
			f.WriteLine("0.0"); //deltaWithoutLearningMomentum
			f.WriteLine(INITIAL_LEARNING_RATE);
		}		
	}

	f.WriteLine("0"); //epoch must be 0
	f.WriteLine("0"); //rmsInterval
	f.WriteLine("0"); //trainingTime
}

int main(int argc, char* argv[]) {
	// Initialize the device
	CudaDevice device;

	if(!device.SupportsCuda()) {
		cout << "Device does not support cuda" << endl;
		return 0;
	}

	device.ShowInfo();

	// Initialize the random generator
	unsigned randomGenerator = (unsigned) time(0);
	srand(randomGenerator);
	cout << "Random Generator.........: " << randomGenerator << endl;

	// Generate the two spirals training dataset
	HostMatrix<cudafloat> twoSpiralsInputs(SPIRAL_POINTS, 2);
	HostMatrix<cudafloat> twoSpiralsOutput(SPIRAL_POINTS, 1);
	GenerateTwoSpiralsDataset(twoSpiralsInputs, twoSpiralsOutput);

	// Create the neural network
	HostArray<int> sizeLayers(4);
	HostArray<bool> selectiveNeurons(3); // The input layer cannot have selective input neurons
	HostArray<int> additionalSpaceLayers; // A single space layer is usually the best option, so no additional space layers will be specified

	sizeLayers[0] = 2; // inputs
	sizeLayers[1] = 30; // 1th hidden layer
	sizeLayers[2] = 10; // 2th hidden layer
	sizeLayers[3] = 1; // output layer

	selectiveNeurons[0] = true; // the 1th hidden layer will have selective activation neurons
	selectiveNeurons[1] = false; // the 2th hidden layer will NOT have selective activation neurons
	selectiveNeurons[2] = false; // the output layer will NOT have selective activation neurons

	MultipleBackPropagation mbp(sizeLayers, selectiveNeurons, additionalSpaceLayers, twoSpiralsInputs, twoSpiralsOutput, INITIAL_LEARNING_RATE);

	// Set the network training parameters
	mbp.SetRobustLearning(true);
	mbp.SetRobustFactor(CUDA_VALUE(0.5));
	mbp.SetMaxPercentageRMSGrow(CUDA_VALUE(0.001));

	mbp.SetMaxStepSize(CUDA_VALUE(10.0));

	mbp.SetUpStepSizeFactor(CUDA_VALUE(1.1));
	mbp.SetDownStepSizeFactor(CUDA_VALUE(0.9));

	mbp.SetMomentum(CUDA_VALUE(0.7));

	// train the network
	cout << endl << "This is a simple example that demonstrates how to use the Multiple Back-Propagation algorithm." << endl;
	cout << "Use ATS (Autonomous Training System) if you want to train other networks." << endl;	
	cout << endl << "training..." << endl;

	do {
		mbp.TrainOneEpoch();
	} while (mbp.GetRMSestimate() > RMS_STOP);

	cout << "Epochs: " << mbp.GetEpoch() << endl;
	cout << "RMS: " << mbp.GetRMS();

	// save the network
	ostringstream sstream;
	sstream << randomGenerator << ".bpn";
	SaveNetwork(sstream.str().c_str(), mbp);

	cout << endl << "A network was saved (" << sstream.str() << "). Use the Multiple Back-Propagation Software, available at http://dit.ipg.pt/MBP, to read it and verify the training results.";
	
	return 0;
}
