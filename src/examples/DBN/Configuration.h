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

#ifndef Configuration_h
#define Configuration_h

#define ERROR_STOP CUDA_VALUE(0.01)
#define DEFAULT_LEARNING_RATE CUDA_VALUE(0.1)
#define DEFAULT_LEARNING_RATE_CLASSIFICATION CUDA_VALUE(0.01)

#include <time.h>

using namespace std;
using namespace GPUMLib;

#include "DataInputFile.h"

class Configuration {
	private:
		int cd;
		bool classification;
		bool hasInvalidParameters;
		bool quiet;
		bool headers;
		bool gpu;
		unsigned randomGenerator;
		HostArray<int> layerNeurons;
		long patterns;
		long testPatterns;
		long maxEpochs;
		long maxEpochsClassification;
		cudafloat mseStop;
		cudafloat rmsStopClassification;
		cudafloat learningRate;
		cudafloat learningRateClassification;
		cudafloat momentum;
		int minibatch;

		string trainFilename;
		string testFilename;

		HostArray<cudafloat> minO;
		HostArray<cudafloat> maxO;

		HostMatrix<cudafloat> inputs;
		HostMatrix<cudafloat> desiredOutputs;
		HostMatrix<cudafloat> testInputs;
		HostMatrix<cudafloat> testDesiredOutputs;

		bool loadedData;
		bool loadedTestData;

		void LoadTrainingData() {
			loadedData = false;

			int nLayers = layerNeurons.Length();
			int nInputs = layerNeurons[0];
			int nOutputs = (classification) ? layerNeurons[nLayers - 1] : 0;

			if (nInputs <= 0 || nOutputs < 0 || patterns <= 0) return;

			minO.ResizeWithoutPreservingData(nOutputs);
			maxO.ResizeWithoutPreservingData(nOutputs);

			inputs.ResizeWithoutPreservingData(patterns, nInputs);
			desiredOutputs.ResizeWithoutPreservingData(patterns, nOutputs);

			DataInputFile f(trainFilename);

			if (!f.GetStream()) return;

			if (headers) f.IgnoreLine();

			int p = 0;

			while (p < patterns && !f.eof()) {
				int i = 0;
				while(i < nInputs && !f.eof()) {
					cudafloat v = f.GetNextValue(false);
					if (v != CUDA_VALUE(0.0) && v != CUDA_VALUE(1.0)) break;
					inputs(p, i) = v;

					i++;
				}

				if (i < nInputs) break;

				int o = 0;
				while(o < nOutputs && !f.eof()) {
					cudafloat v = f.GetNextValue((o + 1) == nOutputs);

					desiredOutputs(p, o) = v;

					if (p == 0) {
						maxO[o] = minO[o] = v;
					} else if (v < minO[o]) {
						minO[o] = v;
					} else if (v > maxO[o]) {
						maxO[o] = v;
					}

					o++;
				}

				if (o < nOutputs) break;

				p++;
			}

			f.Close();

			if(p < patterns) return;

			// rescale
			for(int p = 0; p < patterns; p++) {
				for (int o = 0; o < nOutputs; o++) {
					desiredOutputs(p, o) = (minO[o] == maxO[o]) ? CUDA_VALUE(1.0) : (desiredOutputs(p, o) - minO[o]) / (maxO[o] - minO[o]);
				}
			}

			loadedData = true;
		}

		void LoadTestData() {
			int nLayers = layerNeurons.Length();
			int nInputs = layerNeurons[0];
			int nOutputs = (classification) ? layerNeurons[nLayers - 1] : 0;

			loadedTestData = false;
			if (testPatterns <= 0 || testFilename == "") return;

			testInputs.ResizeWithoutPreservingData(testPatterns, nInputs);
			testDesiredOutputs.ResizeWithoutPreservingData(testPatterns, nOutputs);

			DataInputFile f(testFilename);
			if (!f.GetStream()) return;

			if (headers) f.IgnoreLine();

			int p = 0;
			while (p < testPatterns && !f.eof()) {
				int i = 0;
				while(i < nInputs && !f.eof()) {
					cudafloat v = f.GetNextValue(false);
					if (v != CUDA_VALUE(0.0) && v != CUDA_VALUE(1.0)) break;
					testInputs(p, i) = v;

					i++;
				}

				if (i < nInputs) break;

				int o = 0;
				while(o < nOutputs && !f.eof()) {
					cudafloat v = f.GetNextValue((o + 1) == nOutputs);
					v = (minO[o] == maxO[o]) ? CUDA_VALUE(1.0) : (v - minO[o]) / (maxO[o] - minO[o]);
					testDesiredOutputs(p, o) = v;

					o++;
				}

				if (o < nOutputs) break;

				p++;
			}

			f.Close();

			if(p < testPatterns) return;

			loadedTestData = true;
		}

		char Option(char * p) const {
			if (strlen(p) != 2 || p[0] != '-') return '\0';
			return p[1];
		}

		int ProcessParameter(int argc, char* argv[], int p) {
			int parametersProcessed = 0;

			switch(Option(argv[p])) {
				case 'b':
				case 'B':
					if (++p < argc) {
						minibatch = atoi(argv[p]);
						parametersProcessed = 2;
					}
					break;
				case 'c':
				case 'C':
					classification = true;
					parametersProcessed = 1;
					break;

				case 'e':
				case 'E':
					if (++p < argc) {
						maxEpochs = atol(argv[p]);
						parametersProcessed = 2;

						if (++p < argc && Option(argv[p]) == '\0') {
							maxEpochsClassification = atol(argv[p]);
							parametersProcessed++;
						}
					}
					break;

				case 'g':
				case 'G':
					gpu = false;
					parametersProcessed = 1;
					break;

				case 'h':
				case 'H':
					headers = true;
					parametersProcessed = 1;
					break;

				case 'k':
				case 'K':
					if (++p < argc) {
						cd = atoi(argv[p]);
						parametersProcessed = 2;
					}
					break;

				case 'l':
				case 'L':
					if (++p < argc) {
						learningRate = (cudafloat) atof(argv[p]);
						parametersProcessed = 2;

						if (++p < argc && Option(argv[p]) == '\0') {
							learningRateClassification = (cudafloat) atof(argv[p]);
							parametersProcessed++;
						}
					}
					break;

				case 'm':
				case 'M':
					if (++p < argc) {
						momentum = (cudafloat) atof(argv[p]);
						parametersProcessed = 2;
					}
					break;

				case 'p':
				case 'P':
					if (++p < argc) {
						patterns = atol(argv[p]);
						parametersProcessed = 2;

						if (++p < argc && Option(argv[p]) == '\0') {
							testPatterns = atol(argv[p]);
							parametersProcessed++;
						}
					}
					break;

				case 'q':
				case 'Q':
					quiet = true;
					parametersProcessed = 1;
					break;

				case 'r':
				case 'R':
					if (++p < argc) {
						randomGenerator = atol(argv[p]);
						parametersProcessed = 2;
					}
					break;

				case 's':
				case 'S':
					if (++p < argc) {
						rmsStopClassification = mseStop = (cudafloat) atof(argv[p]);
						parametersProcessed = 2;

						if (++p < argc && Option(argv[p]) == '\0') {
							rmsStopClassification = (cudafloat) atof(argv[p]);
							parametersProcessed++;
						}
					}
					break;

				case 't':
				case 'T':
					if (++p < argc) {
						char * topology = argv[p];
						size_t lenght = strlen(topology);

						int numberLayers = 0;
						bool isValid = true;
						bool layerContainsNeurons = false;
						for (size_t i = 0; i < lenght && isValid; i++) {
							char c = topology[i];

							if (c == '-') {
								if (layerContainsNeurons) {
									layerContainsNeurons = false;
									++numberLayers;
								} else {
									isValid = false;
								}
							} else if (c >= '1' && c <= '9') {
								layerContainsNeurons = true;
							} else if (c != '0') {
								isValid = false;
							}
						}

						if (layerContainsNeurons) {
							++numberLayers;
						} else {
							isValid = false;
						}

						if (numberLayers < 2) isValid = false;

						if (isValid) {
							layerNeurons.ResizeWithoutPreservingData(numberLayers);

							int layer = 0;
							int neurons = 0;

							for (size_t i = 0; i < lenght; i++) {
								char c = topology[i];

								if (c == '-') {
									layerNeurons[layer++] = neurons;
									neurons = 0;
								} else {
									neurons = neurons * 10 + (c - '0');
								}
							}

							layerNeurons[layer] = neurons;

							parametersProcessed = 2;
						}
					}
					break;
			}

			return parametersProcessed;
		}

	public:
		Configuration(int argc, char* argv[]) {
		    minibatch = 0;
			cd = 1;
			classification = false;
			hasInvalidParameters = false;
			quiet = false;
			headers = false;
			gpu = true;
			randomGenerator = (unsigned int) time(0);
			patterns = 0;
			testPatterns = 0;
			maxEpochs = maxEpochsClassification = 0;
			rmsStopClassification = mseStop = ERROR_STOP;
			learningRate = DEFAULT_LEARNING_RATE;
			learningRateClassification = DEFAULT_LEARNING_RATE_CLASSIFICATION;
			momentum = DEFAULT_MOMENTUM;
			loadedData = false;

			if (argc < 2) {
				hasInvalidParameters = true;
				return;
			}

			trainFilename = argv[1];

			if (argc > 2) {
				int processedParameters = 2;

				if (Option(argv[2]) == '\0') {
					testFilename = argv[2];
					processedParameters++;
				}

				for(int p = processedParameters; p < argc; p += processedParameters) {
					processedParameters = ProcessParameter(argc, argv, p);

					if (processedParameters == 0) {
						hasInvalidParameters = true;
						return;
					}
				}
			}

			if (layerNeurons.Length() < 2) hasInvalidParameters = true;

			if (!hasInvalidParameters) {
				LoadTrainingData();

				if (testPatterns > 0 && classification) {
					LoadTestData();
				} else {
					loadedTestData = true;
				}
			}

			srand(randomGenerator);

			//network = nullptr;
			//if (loadedData) CreateHostRBMNetwork();
		}

		bool Quiet() const {
			return quiet;
		}

		bool HasInvalidParameters() const {
			return hasInvalidParameters;
		}

		bool LoadedTrainData() const {
			return loadedData;
		}

		bool LoadedTestData() const {
			return loadedTestData;
		}

		void ShowParameterInfo() const {
			cout << "usage:" << endl << "DBN <train data file> [<test data file>] -t <topology> -p <number of training samples> [<number of test samples>] [-h] [-s <stop MSE> [<stop RMS classification>]] [-e <maximum number of epochs> [<maximum number of epochs for classification>]] [-r <random generator>] [-q] [-c] [-k <value>] [-l <initial learning rate> [<initial learning rate for classification>]] [-m <momentum>] [-b <mini bactch size>]" << endl << endl;

			cout << "-t : topology (Example: -t 10-30-10-1). At least one input and one output layer must be specified" << endl;

			cout << "-g : Do not use the GPU. Use CPU instead." << endl;
			cout << "-h : The training and test files have a header (default: no)" << endl;
			cout << "-k : Use CD-k" << endl;
			cout << "-c : Classification (After the unsupervised training the network will be trained with the BP algorithm)" << endl;
			cout << "-q : Quiet" << endl;
			cout << "-s : Stop Errors (default is " << ERROR_STOP << ")" << endl;
		}

		HostArray<int> & Layers() {
			return layerNeurons;
		}

		HostMatrix<cudafloat> & Inputs() {
			return inputs;
		}

		HostMatrix<cudafloat> & TestInputs() {
			return testInputs;
		}

		HostMatrix<cudafloat> & DesiredOutputs() {
			return desiredOutputs;
		}

		HostMatrix<cudafloat> & DesiredTestOutputs() {
			return testDesiredOutputs;
		}

		long MaximumEpochs() const {
			return maxEpochs;
		}

		long MaximumEpochsClassification() const {
			return maxEpochsClassification;
		}

		unsigned RandomGenerator() const {
			return randomGenerator;
		}

		void ResetRandomGenerator() {
			srand(++randomGenerator);
		}

		string Topology() const {
			ostringstream s;

			s << layerNeurons[0];

			int layers = layerNeurons.Length();
			for(int l = 1; l < layers; l++) s << '-' <<  layerNeurons[l];

			return s.str();
		}

		string TrainDataFile() const {
			return trainFilename;
		}

		string TestDataFile() const {
			return testFilename;
		}

		int NumberTestPatterns() const {
			return (loadedTestData) ? testPatterns : 0;
		}

		cudafloat StopMSE() const {
			return mseStop;
		}

		cudafloat RmsStopClassification() const {
			return rmsStopClassification;
		}

		int CD() const {
			return cd;
		}

		bool Classification() const {
			return classification;
		}

		bool UseGPU() const {
			return gpu;
		}

		cudafloat LearningRate() const {
			return learningRate;
		}

		cudafloat LearningRateClassification() const {
			return learningRateClassification;
		}

		cudafloat Momentum() const {
			return momentum;
		}

		int MiniBatchSize() const {
		    return minibatch;
		}
};

#endif