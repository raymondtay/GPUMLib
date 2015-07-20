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

#include <time.h>
#include <sstream>
#include <stdlib.h>
#include <math.h>

using namespace GPUMLib;
using namespace std;

#include "../../memory/HostArray.h"
#include "../../common/CudaDefinitions.h"
#include "DataInputFile.h"

class Configuration {
	private:
		bool hasInvalidParameters;
		long numberNetworksTrain;
		bool quiet;
		bool headers;
		unsigned randomGenerator;
		bool mbp;
		HostArray<int> layerNeurons;
		int patterns;
		int testPatterns;
		long maxEpochs;
		bool fixedTopology;
		cudafloat rmsStop;
		bool robustLearning;
		bool rescale;

		string trainFilename;
		string testFilename;

		HostArray<cudafloat> minI;
		HostArray<cudafloat> maxI;
		HostArray<cudafloat> minO;
		HostArray<cudafloat> maxO;

		bool hasMissingValues;
		int columnInvalidTestMissingValues;

		HostArray<bool> varContainsMissingValues;

		HostMatrix<cudafloat> inputs;
		HostMatrix<cudafloat> desiredOutputs;
		HostMatrix<cudafloat> testInputs;
		HostMatrix<cudafloat> testDesiredOutputs;

		bool loadedData;
		bool loadedTestData;

		cudafloat AbsDiff(cudafloat a, cudafloat b) {
			cudafloat d = a - b;
			if (d < CUDA_VALUE(0.0)) return -d;		
			return d;
		}

		void LoadTrainingData() {
			loadedData = false;

			int nLayers = layerNeurons.Length();
			int nInputs = layerNeurons[0];
			int nOutputs = layerNeurons[nLayers - 1];

			if (nInputs <= 0 || nOutputs <= 0 || patterns <= 0) return;

			varContainsMissingValues.ResizeWithoutPreservingData(nInputs);

			minI.ResizeWithoutPreservingData(nInputs);
			maxI.ResizeWithoutPreservingData(nInputs);
			minO.ResizeWithoutPreservingData(nOutputs);
			maxO.ResizeWithoutPreservingData(nOutputs);

			inputs.ResizeWithoutPreservingData(patterns, nInputs);
			desiredOutputs.ResizeWithoutPreservingData(patterns, nOutputs);

			DataInputFile f(trainFilename);

			if (!f.GetStream()) return;

			if (headers) f.IgnoreLine();

			for(int i = 0; i < nInputs; i++) {
				minI[i] = numeric_limits<cudafloat>::max();
				maxI[i] = numeric_limits<cudafloat>::min();

				varContainsMissingValues[i] = false;
			}

			int p = 0;

			while (p < patterns && !f.eof()) {
				int i = 0;
				while(i < nInputs && !f.eof()) {
					cudafloat v = f.GetNextValue(false);

					inputs(p, i) = v;

					if (IsInfOrNaN(v)) {
						hasMissingValues = true;
						varContainsMissingValues[i] = true;
					} else {
						if (v < minI[i]) minI[i] = v;
						if (v > maxI[i]) maxI[i] = v;
					}

					i++;
				}

				if (i < nInputs) {
					cout << "Could not load all inputs (" << i << " inputs read in sample " << (p + 1) << ")" << endl;
					break;
				}

				int o = 0;
				while(o < nOutputs && !f.eof()) {
					cudafloat v = f.GetNextValue((o + 1) == nOutputs);

					desiredOutputs(p, o) = v;
					
					if (IsInfOrNaN(v)) {
						break;
					} else {
						if (p == 0) {
							maxO[o] = minO[o] = v;
						} else if (v < minO[o]) {
							minO[o] = v;
						} else if (v > maxO[o]) {
							maxO[o] = v;
						}
					}

					o++;
				}

				if (o < nOutputs) {
					cout << "Could not load all outputs (" << o << " outputs read in sample " << (p + 1) << ", expected " << nOutputs << ")" << endl;
					break;
				}

				p++;
			}

			f.Close();

			if(p < patterns) {
				cout << "Could not load all samples (" << p << " samples read)" << endl;
				return;
			}

			bool warning = false;

			for (int i = 0; i < nInputs; i++) {
				if (AbsDiff(maxI[i], minI[i]) < CUDA_VALUE(0.0000001)) {
					maxI[i] = minI[i];

					if (!warning) {
						cout << "WARNING: The following columns may present problems to the training process (variables are allways constant)" << endl;
						warning = true;
					}

					int div = i / 26;
					char c = 'A' + (div - 1);
					cout << (i + 1) << "(";
					if (div > 0) cout << c;
					c = 'A' + (i - div * 26);
					cout << c << ")" << endl;
				}
			}

			if (warning) cout << endl;

			if (rescale) {
				for(int p = 0; p < patterns; p++) {
					for (int i = 0; i < nInputs; i++) {
						cudafloat v = inputs(p, i);

						if (IsInfOrNaN(v)) {
							inputs(p, i) = v; //MISSING_VALUE;
						} else {
							inputs(p, i) = (minI[i] == maxI[i]) ? CUDA_VALUE(1.0) : (CUDA_VALUE(-1.0) + CUDA_VALUE(2.0) * (v - minI[i]) / (maxI[i] - minI[i]));
						}
					}

					for (int o = 0; o < nOutputs; o++) {
						cudafloat doutput;

						if (minO[o] == maxO[o]) {
							doutput = (minO[o] >= CUDA_VALUE(0.5)) ? CUDA_VALUE(1.0) : CUDA_VALUE(0.0);
						} else {
							doutput = (desiredOutputs(p, o) - minO[o]) / (maxO[o] - minO[o]);
						}

						desiredOutputs(p, o) = doutput;
					}
				}
			}

			loadedData = true;
		}

		void LoadTestData() {
			int nLayers = layerNeurons.Length();
			int nInputs = layerNeurons[0];
			int nOutputs = layerNeurons[nLayers - 1];

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

					if (IsInfOrNaN(v)) {
						hasMissingValues = true;
						if (!varContainsMissingValues[i]) {
							columnInvalidTestMissingValues = i;
							break;
						}
					} else if (rescale) {
						v = (minI[i] == maxI[i]) ? CUDA_VALUE(1.0) : (CUDA_VALUE(-1.0) + CUDA_VALUE(2.0) * (v - minI[i]) / (maxI[i] - minI[i]));
					}

					testInputs(p, i) = v;

					i++;
				}

				if (i < nInputs) break;

				int o = 0;
				while(o < nOutputs && !f.eof()) {
					cudafloat v = f.GetNextValue((o + 1) == nOutputs);

					if (rescale) {
						if (minO[o] == maxO[o]) {
							v = (v >= CUDA_VALUE(0.5)) ? CUDA_VALUE(1.0) : CUDA_VALUE(0.0);
						} else {
							v = (v - minO[o]) / (maxO[o] - minO[o]);
						}
					}
					
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
					mbp = false;
					parametersProcessed = 1;
					break;

				case 'd':
				case 'D':
					rescale = false;
					parametersProcessed = 1;
					break;

				case 'e':
				case 'E':
					if (++p < argc) {
						maxEpochs = atol(argv[p]);
						parametersProcessed = 2;
					}
					break;

				case 'f':
				case 'F':
					fixedTopology = true;
					parametersProcessed = 1;
					break;

				case 'h':
				case 'H':
					headers = true;
					parametersProcessed = 1;
					break;

				case 'n':
				case 'N':
					if (++p < argc) {
						numberNetworksTrain = atol(argv[p]);
						parametersProcessed = 2;
					}
					break;

				case 'p':
				case 'P':
					if (++p < argc) {
						patterns = atoi(argv[p]);
						parametersProcessed = 2;

						if (++p < argc && Option(argv[p]) == '\0') {
							testPatterns = atoi(argv[p]);
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
					if (++p < argc) {
						randomGenerator = atol(argv[p]);
						parametersProcessed = 2;
					}
					break;

				case 'R':
					robustLearning = false;
					parametersProcessed = 1;
					break;

				case 's':
				case 'S':
					if (++p < argc) {
						rmsStop = (cudafloat) atof(argv[p]);
						parametersProcessed = 2;
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
			hasInvalidParameters = false;
			numberNetworksTrain = 1;
			quiet = false;			
			headers = false;
			randomGenerator = (unsigned int) time(0);
			patterns = 0;
			testPatterns = 0;
			mbp = true;
			maxEpochs = 0;
			fixedTopology = false;
			rmsStop = RMS_STOP;			
			loadedData = false;
			hasMissingValues = false;
			columnInvalidTestMissingValues = -1;
			robustLearning = true;
			rescale = true;

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

			if (!hasInvalidParameters) {
				LoadTrainingData();			

				if (testPatterns > 0) {
					LoadTestData();
				} else {
					loadedTestData = true;
				}
			}

			srand(randomGenerator);

			if (loadedData && columnInvalidTestMissingValues >= 0) {
				cout << "The test dataset contains columns with missing values that do not appear in the training dataset (column " << (columnInvalidTestMissingValues + 1) << ")." <<  endl;
				loadedData = false;
			}
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
			cout << "usage:" << endl << "ATS <train data file> [<test data file>] -t <topology> -p <number of training samples> [<number of test samples>] [-h] [-n <number networks to train>] [-s <stop RMS>] [-e <maximum number of epochs>] [-r <random generator>] [-R] [-b] [-f] [-q] [-d]" << endl << endl;

			cout << "-t : topology (Example: -t 10-30-10-1). At least one input and one output layer must be specified" << endl;

			cout << "-h : The training and test files have a header (default: no)" << endl;
			cout << "-b : BP algoritm (default: MBP)" << endl;
			cout << "-f : Fixed topology" << endl;
			cout << "-q : Quiet" << endl;
			cout << "-R : do not use robust learning" << endl;
			cout << "-s : Stop RMS (default is " << RMS_STOP << ")" << endl;
			cout << "-d : Do not perform data rescale" << endl;
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
		
		long MaximumEpochs() {
			return maxEpochs;
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

		const char * Network() const {
			return (mbp) ? "MBP " : "BP ";
		}

		bool MBP() {
			return mbp;
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
		
		cudafloat RmsStop() {
			return rmsStop;
		}

		int NumberNetworksTrain() const {
			return numberNetworksTrain;
		}

		bool FixedTopology() const {
			return fixedTopology || layerNeurons.Length() <= 2;
		}

		bool RobustLearning() const {
			return robustLearning;
		}
};

#endif
