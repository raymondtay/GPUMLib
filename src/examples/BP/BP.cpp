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
#include "../../MBP/BackPropagation.h"
#include "../MBP/MBPutils.h"

#define SPIRAL_DENSITY (1)
#define SPIRAL_POINTS (2 * (96 * SPIRAL_DENSITY + 1))

#define MAX_RADIANS 6.5

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

	sizeLayers[0] = 2; // inputs
	sizeLayers[1] = 30; // 1th hidden layer
	sizeLayers[2] = 10; // 2th hidden layer
	sizeLayers[3] = 1; // output layer

	BackPropagation bp(sizeLayers, twoSpiralsInputs, twoSpiralsOutput, INITIAL_LEARNING_RATE);

	// Set the network training parameters
	bp.SetRobustLearning(true);
	bp.SetRobustFactor(CUDA_VALUE(0.5));
	bp.SetMaxPercentageRMSGrow(CUDA_VALUE(0.001));

	bp.SetMaxStepSize(CUDA_VALUE(10.0));

	bp.SetUpStepSizeFactor(CUDA_VALUE(1.1));
	bp.SetDownStepSizeFactor(CUDA_VALUE(0.9));

	bp.SetMomentum(CUDA_VALUE(0.7));

	// train the network
	cout << endl << "This is a simple example that demonstrates how to use the Back-Propagation algorithm." << endl;
	cout << "Use ATS (Autonomous Training System) if you want to train other networks." << endl;
	cout << endl << "training..." << endl;

	do {
		bp.TrainOneEpoch();
	} while (bp.GetRMSestimate() > RMS_STOP);

	cout << "Epochs: " << bp.GetEpoch() << endl;
	cout << "RMS: " << bp.GetRMS();
	
	// save the network
	ostringstream sstream;
	sstream << randomGenerator << ".bpn";
	SaveMBPNetwork(sstream.str().c_str(), &bp, "twospirals.txt", "", RMS_STOP, 1000000, false);

	cout << endl << "A network was saved (" << sstream.str() << "). Use the Multiple Back-Propagation Software, available at http://dit.ipg.pt/MBP, to read it and verify the training results.";
	
	return 0;
}