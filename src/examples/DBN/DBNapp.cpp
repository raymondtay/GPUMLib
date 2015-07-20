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

#include "../../MBP/BackPropagation.h"
#include "../../random/random.h"
#include "../../DBN/HostDBN.h"
#include "../../DBN/DBN.h"
#include "../common/CudaInit.h"
#include "Configuration.h"
#include "OutputFile.h"
#include "../common/ConfusionMatrix.h"
#include "../MBP/MBPutils.h"

using namespace GPUMLib;

void Save(OutputFile & f, float v) {
	f.Write("<float>");
	f.Write(((double) v));
	f.WriteLine("</float>");
}

void SaveDBNheader(OutputFile & f) {
	f.WriteLine("<?xml version=\"1.0\" encoding=\"utf-8\"?>");
	f.WriteLine("<DBNmodel xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\">");
	f.WriteLine("<layers>");
}

void SaveDBNfooter(OutputFile & f, Configuration & cfg) {
	f.WriteLine("</layers>");

	f.Write("<TrainFilename>");
	f.Write(cfg.TrainDataFile().c_str());
	f.Write("</TrainFilename>");

	f.Write("<TestFilename>");
	f.Write(cfg.TestDataFile().c_str());
	f.Write("</TestFilename>");

	f.WriteLine("</DBNmodel>");
}

void SaveDBNlayer(OutputFile & f, HostMatrix<cudafloat> & weights, HostArray<cudafloat> & a, HostArray<cudafloat> & b) {
	int I = weights.Columns();
	int J = weights.Rows();

	f.WriteLine("<RBMlayer>");

	f.WriteLine("<weights>");
	for(int j = 0; j < J; j++) for(int i = 0; i < I; i++) Save(f, weights(j, i));
	f.WriteLine("</weights>");

	f.WriteLine("<biasVisibleLayer>");
	for(int i = 0; i < I; i++) Save(f, a[i]);
	f.WriteLine("</biasVisibleLayer>");

	f.WriteLine("<biasHiddenLayer>");
	for(int j = 0; j < J; j++) Save(f, b[j]);
	f.WriteLine("</biasHiddenLayer>");

	f.WriteLine("</RBMlayer>");
}

void SaveDBN(DBNhost & network, Configuration & cfg) {
	ostringstream sstream;
	sstream << cfg.RandomGenerator() << ".dbn";

	OutputFile f(sstream.str().c_str());
	string s;

	SaveDBNheader(f);

	int nLayers = network.GetNumberRBMs();
	for(int l = 0; l < nLayers; l++) {
		RBMhost * layer = network.GetRBM(l);

		HostMatrix<cudafloat> w = layer->GetWeights();
		HostArray<cudafloat> a = layer->GetVisibleBias();
		HostArray<cudafloat> b = layer->GetHiddenBias();

		SaveDBNlayer(f, w, a, b);
	}

	SaveDBNfooter(f, cfg);
}

void SaveDBN(DBN & network, Configuration & cfg) {
	ostringstream sstream;
	sstream << cfg.RandomGenerator() << ".dbn";

	OutputFile f(sstream.str().c_str());
	string s;

	SaveDBNheader(f);

	int nLayers = network.GetNumberRBMs();
	for(int l = 0; l < nLayers; l++) {
		RBM * layer = network.GetRBM(l);

		HostMatrix<cudafloat> w = layer->GetWeights();
		HostArray<cudafloat> a = layer->GetVisibleBias();
		HostArray<cudafloat> b = layer->GetHiddenBias();

		SaveDBNlayer(f, w, a, b);
	}

	SaveDBNfooter(f, cfg);
}

int main(int argc, char * argv[]) {
	// Load information and check parameters 
	Configuration cfg(argc, argv);

	if (cfg.HasInvalidParameters()) {
		cout << endl << "Invalid parameter or no parameters specified. ";
		cfg.ShowParameterInfo();
		return 0;
	}

	if (!cfg.LoadedTrainData()) {
		cout << "Could not load the training data. Check the filename and the number of inputs, outputs and samples. The file must contain only binary inputs." << endl;
		return 0;
	}

	if (!cfg.LoadedTestData()) {
		cout << "Could not load the test data. Check the filename and the number of inputs, outputs and samples. The file must contain only binary inputs." << endl;
		return 0;
	}

	HostArray<int> & layers = cfg.Layers();

	int numberLayers = layers.Length();
	if (cfg.Classification()) numberLayers--;

	if (numberLayers < 2) {
		cout << "Invalid topology. The network must have at least 2 layers (3 if classification is used)." << endl;
		return 0;
	}

	bool useGPU = cfg.UseGPU();

	// Initialize the device
	CudaDevice device;

	if (useGPU && !device.SupportsCuda()) {
		cout << "The device does not support cuda." << endl;
		useGPU = false;
	}	

	if (!cfg.Quiet()) {
		if (useGPU) {
			device.ShowInfo();
		} else {
			cout << "Using CPU " << endl;
		}

		cout << "Random Generator.........: " << cfg.RandomGenerator() << endl;
		cout << "Network..................: " << cfg.Topology() << endl << endl;
	}

	HostArray<int> dbnLayers(numberLayers);
	for(int l = 0; l < numberLayers; l++) dbnLayers[l] = layers[l];

	HostMatrix<cudafloat> & inputs = cfg.Inputs();
	HostMatrix<cudafloat> & desiredOutputs = cfg.DesiredOutputs();
	HostMatrix<cudafloat> & testInputs = cfg.TestInputs();
	HostMatrix<cudafloat> & desiredTestOutputs = cfg.DesiredTestOutputs();

	int numberRBMs = numberLayers - 1;

	clock_t initialTime;
	unsigned time;

	if (useGPU) {
		Random::SetSeed(cfg.RandomGenerator());

		DBN dbn(dbnLayers, inputs, cfg.LearningRate(), cfg.Momentum());

		initialTime = clock();
		if (dbn.Train(cfg.MaximumEpochs(), cfg.CD(), cfg.MiniBatchSize(), cfg.StopMSE())) {
			cudaThreadSynchronize();
			time = (clock() - initialTime);

			for(int l = 0; l < numberRBMs; l++) {
				RBM * layer = dbn.GetRBM(l);
				cout << "Layer " << (l + 1) << " : " << layer->GetMSE() << " after " << layer->Epoch() << " epochs" << endl;
			}
			cout << (double) time / CLOCKS_PER_SEC << " s" << endl;

			SaveDBN(dbn, cfg);
		} else {
			cout << "Could not train the network - Insuficient device (GPU) memory" << endl;
		}
	} else {
		if (cfg.MiniBatchSize() > 0) {
			cout << "Currently mini-batch is only supported by the GPU version. Training will proceed ignoring this option" << endl;
		}

		DBNhost dbn(dbnLayers, inputs, cfg.LearningRate(), cfg.Momentum());

		initialTime = clock();
		dbn.Train(cfg.MaximumEpochs(), cfg.CD(), cfg.StopMSE());
		time = (clock() - initialTime);

		for(int l = 0; l < numberRBMs; l++) {
			RBMhost * layer = dbn.GetRBM(l);
			cout << "Layer " << (l + 1) << " : " << layer->MeanSquareError() << " after " << layer->Epoch() << " epochs" << endl;
		}

		cout << endl << (double) time / CLOCKS_PER_SEC << " s" << endl;
		SaveDBN(dbn, cfg);

		if (cfg.Classification()) {
			cout << "Currently classfication is only supported by the GPU version." << endl;
		}
	}	

	return 0;
}

