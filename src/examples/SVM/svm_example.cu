/*
Joao Goncalves is a MSc Student at the University of Coimbra, Portugal
Copyright (C) 2012 Joao Goncalves

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

//this example implements a SVM using the GPU. Can be used to train and classify binary datasets

#ifdef _WIN32
	#include <windows.h>
#else
	#include <sys/time.h>
#endif

#include <assert.h>
#include <float.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

//GPUMLib stuff
#include "../../common/CudaDefinitions.h"
#include "../../common/Utilities.h"
#include "../../memory/DeviceArray.h"
#include "../../memory/DeviceMatrix.h"
#include "../../memory/HostArray.h"
#include "../../memory/HostMatrix.h"
#include <cuda.h>

//! Comment or set this macro to zero to disable some runtime debugging info
#define DEBUG 1

#include "../../SVM/Settings.h"
#include "../../SVM/svm_kernel_type.h"

//SVM encapsulating class
#include "../../SVM/SVM.h"

using namespace std;

namespace GPUMLib {

	typedef unsigned int uint;

	//! Value separator tag for the CSV files
#define VALUE_SEPARATOR ",;"

	//! Size of blocks (in elements) for reading/writing operations
#define BUFFERING_BLOCK_SIZE (1<<24)

#ifndef BUFFERING_BLOCK_SIZE
#define BUFFERING_BLOCK_SIZE (1<<20)
#endif

	/**
	* helper function for validLine()
	*/
	bool validCharacter(char &c) {
		if (c >= '0' && c <= '9')
			return true;
		if (c == ',' || c == '.' || c == ' ' || c == '\r' || c == '\n' || c == '-' || c == 'e' || c == 'E')
			return true;
		return 0;
	}

	/**
	* check if a line from the dataset is valid
	* a line is valid if it only contains valid characters
	*/
	bool validLine(char * buf, int size) {
		for (int i = 0; i < size; i++) {
			char c = buf[i];
			if (c == 0)
				return true;
			if (!validCharacter(c))
				return false;
		}
		return true;
	}

	/**
	* Counts the amount of samples (lines) in given comma separated file (CSV) file
	* @param f The file to be used
	* @return The number of samples in the file
	*/
	int getNumberOfSamples(FILE *f) {
		//TODO: remove that dirty way using valid characters (because of e/E)
		//start... from the beginning
		fseek(f, 0, SEEK_SET);

		//read
		char * buf = new char[BUFFERING_BLOCK_SIZE];
		int count = 0;
		while (fgets(buf, BUFFERING_BLOCK_SIZE, f)) {
			if (validLine(buf, BUFFERING_BLOCK_SIZE))
				count++;
		}
		//	if (DEBUG)
		//		cout << "Number of samples:\t" << count << endl;
		free(buf);
		return count;
	}

	/**
	* Counts the amount of lines (\n) in given file
	* @param f The file to be used
	* @return The number of lines in the file
	*/
	int getNumberOfLines(FILE *f) {
		int count = 0;

		//start from the beginning
		fseek(f, 0, SEEK_SET);

		//read
		char * buf = new char[BUFFERING_BLOCK_SIZE];
		for (;;) {
			//read a nice chunk (to minimize head seek overhead)
			size_t amount_read = fread(buf, sizeof(char), BUFFERING_BLOCK_SIZE, f);
			if (amount_read == 0)
				break;
			//count occurrences of '\n' in that chunk
			for (size_t i = 0; i < amount_read; i++) {
				if (buf[i] == '\n')
					count++;
			}
		}
		free(buf);
		//	if (DEBUG)
		//		cout << "Number of lines:\t" << count << endl;
		return count;
	}

	/**
	* Counts the amount of columns in first line of given CSV file.
	* @param f The file to be used
	* @return The number of columns in the file
	*/
	int getNumberOfColumns(FILE *f) {
		//start from the beginning
		fseek(f, 0, SEEK_SET);

		//temporary storage
		char * buf = new char[BUFFERING_BLOCK_SIZE];

		//eat empty lines
		bool gotvalidline = false;
		while (!gotvalidline) {
			fgets(buf, BUFFERING_BLOCK_SIZE, f);
			if (buf[0] != '\n' && validLine(buf, BUFFERING_BLOCK_SIZE))
				gotvalidline = true;
		}

		//eat first value
		char* tok = strtok(buf, VALUE_SEPARATOR);
		int num_columns = 1;
		//count next values until the end of the line
		while ((tok = strtok(NULL, VALUE_SEPARATOR)) != NULL)
			num_columns++;

		//	if (DEBUG)
		//		cout << "Number of columns:\t" << num_columns << endl;
		return num_columns;
	}


	/**
	* Reads a CSV file as a dataset.
	* @param f The file to be read
	* @param samples The HostMatrix where to store the attributes/features for each sample/pattern
	* @param classes The HostArray where to store the class of each sample/pattern
	* @param ncols The number of columns in the CSV file (must be previously obtained)
	* @param positive_class The value to be used to identify the positive class. The other values will be used as the negative class
	*/
	void readDataSet(FILE *f, GPUMLib::HostMatrix<cudafloat> & samples, GPUMLib::HostArray<int> & classes, int ncols, int positive_class) {
		//start from the beginning
		fseek(f, 0, SEEK_SET);

		//read
		char * buf = new char[BUFFERING_BLOCK_SIZE];
		int row = 0;
		int positives = 0;
		int negatives = 0;
		while (fgets(buf, BUFFERING_BLOCK_SIZE, f)) {
			if (!validLine(buf, BUFFERING_BLOCK_SIZE))
				continue;
			//strrplchr(buf, ',', '.'); // replace , by .
			//get first feature and convert to numeric
			char *tok = strtok(buf, VALUE_SEPARATOR);
			double val = strtod(tok, NULL); // atoi IS SLOWER!
			samples(row, 0) = (float) val;
			//do the same for the remaining features
			for (int col = 1; col < ncols - 1; col++) {
				tok = strtok(NULL, VALUE_SEPARATOR);
				val = strtod(tok, NULL);
				// store value
				samples(row, col) = (float) val;
			}
			// get the class
			tok = strtok(NULL, VALUE_SEPARATOR);
			int c = strtol(tok, NULL, 10);
			//we expect the class label to belong to {-1;1}
			if (c == positive_class) {
				classes[row] = 1;
				negatives++;
			} else {
				classes[row] = -1;
				positives++;
			}
			row++;
		}
		if (DEBUG) {
			cout << "read dataset with " << row << " rows and " << ncols << " columns" << endl;
			cout << "dataset with " << positives << " positives and " << negatives << " negatives" << endl;
			cout << "positive to negative ratio is " << (double) positives / (double) negatives << endl;
		}
	}

	//! Prints various classification metrics such as the Confusion Matrix, Accuracy, F-Score, etc.
	//! \param targets A HostArray containing the target classes (real data)
	//! \param predicted A HostArray containing the predicted classes (output of the classifier)
	//! \param length The sample size
	void showClassificationMetrics(GPUMLib::HostArray<int> &targets, GPUMLib::HostArray<int> &predicted, int length) {
		//confusion matrix
		int tp = 0;
		int fp = 0;
		int tn = 0;
		int fn = 0;

		int errors = 0;
		//#pragma omp parallel for reduction(+:errors,tp,fp,tn,fn)
		for (int i = 0; i < length; i++) {
			//confusion matrix
			if (predicted[i] == -1) {
				if (targets[i] == -1) {
					//TN
					tn++;
				} else {
					//FN
					fn++;
				}
			} else {
				if (targets[i] == -1) {
					//FP
					fp++;
				} else {
					//TP
					tp++;
				}
			}
			int class_err = targets[i] - predicted[i];
			if (class_err != 0)
				errors++;
		}
		cout << "Confusion matrix:" << endl;
		cout << "\t\t\tActual class" << endl;
		cout << "\t\t\t-1\t1" << endl;
		cout << "Predicted class\t-1\t" << tn << "\t" << fn << endl;
		cout << "\t\t1\t" << fp << "\t" << tp << endl;

		double precision = ((tp + fp) == 0 ? 0 : (double) (tp) / (double) (tp + fp));
		cout << "Precision: " << precision << endl;
		double recall = ((fn + tp) == 0 ? 0 : (double) (tp) / (double) (fn + tp));
		cout << "Recall: " << recall << endl;
		double false_positive_rate = ((fp + tn) == 0 ? 0 : (double) (fp) / (double) (fp + tn));
		cout << "False Positive Rate: " << false_positive_rate << endl;
		cout << "Specificity: " << 1.0 - false_positive_rate << endl;
		cout << "False Discovery Rate: " << ((fp + tp) == 0?0:(double) (fp) / (double) (fp + tp)) << endl;

		cout << "Accuracy: " << ((tp + tn + fp + fn) == 0?0:(double) (tp + tn) / (double) (tp + tn + fp + fn)) << endl;
		cout << "F-score: " << ((recall + precision) <FLT_MIN?0:(2.0 * recall * precision) / (recall + precision)) << endl;
		cout << "testing errors were " << errors << "/" << length << " = " << (double) errors / (double) length << endl;
	}

	//! Saves the SVM data/model (composed of the features, non-zero alphas (SVs) and the bias) to a file
	//! \param model_filename The filename where to save data to
	//! \param model The HostMatrix containing the non-zero alphas (SVs) and the features
	//! \param b The hyperplane's bias
	void saveModel(char * model_filename, GPUMLib::HostMatrix<cudafloat> &model, cudafloat b) {
		if (DEBUG)
			cout << "saving model to file..." << endl;
		char * WRITE_BUF = new char[BUFFERING_BLOCK_SIZE];
		FILE *model_file;
		model_file = fopen(model_filename, "w");
		if (model_file) {
			//buffer process
			setvbuf(model_file, WRITE_BUF, _IOFBF, BUFFERING_BLOCK_SIZE);
			//first line gives the amount of support vectors
			fprintf(model_file, "%d\n", model.Rows());
			//second line gives the amount of features
			fprintf(model_file, "%d\n", model.Columns() - 2);
			//third line the hyperplane offset
			fprintf(model_file, "%f\n", b);
			//the remaining lines are in the form:
			//alpha_i | class_i | attribute_0 ... attribute_n-1
			for (int sv_i = 0; sv_i < model.Rows(); sv_i++) {
				for (int col = 0; col < model.Columns(); col++) {
					fprintf(model_file, "%f", model(sv_i, col));
					if (col < model.Columns() - 1)
						fprintf(model_file, ",");
				}
				fprintf(model_file, "\n");
			}
			fclose(model_file);
			printf("model saved to file %d\n", model_file);
		} else
			cout << "Err: Unable to open model file for write." << endl;
		delete WRITE_BUF;
	}

	//! Returns the amount of cores per SM depending on its architecture (taken from nvidia's cutil)
	//! \param major The Major revision number of the reported CUDA support
	//! \param minor The Minor revision number of the reported CUDA support
	//! \return The amount of Cuda Cores / SPs
	inline int convertSMVer2Cores(int major, int minor) {
		// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
		typedef struct {
			int SM; // 0xMm (hexadecimal notation), M = SM Major version, and m = SM minor version
			int Cores;
		} sSMtoCores;

		sSMtoCores nGpuArchCoresPerSM[] = 
		{{0x10,  8 },
		{ 0x11,  8 },
		{ 0x12,  8 },
		{ 0x13,  8 },
		{ 0x20, 32 },
		{ 0x21, 48 },
		{   -1, -1 } 
		};

		int index = 0;
		while (nGpuArchCoresPerSM[index].SM != -1) {
			if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
				return nGpuArchCoresPerSM[index].Cores;
			}
			index++;
		}
		printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
		return -1;
	}

	//! Automatically selects the fastest available compute device, if it fails to find one, it automatically aborts execution
	void selectFastestDevice(){
		int num_devices=0;
		int device=0;
		cudaGetDeviceCount(&num_devices);
		cout << "found " << num_devices << " CUDA devices" << endl;
		if(num_devices==cudaErrorNoDevice){
			puts("cudaErrorNoDevice");
		}else if(num_devices==cudaErrorInsufficientDriver){
			puts("cudaErrorInsufficientDriver");
		}
		if (num_devices > 0) {
			int max_multiprocessors = 0;
			int fastest_device = -1;
			//		puts("-------------------");
			for (device = 0; device < num_devices; device++) {
				cudaDeviceProp properties;
				cudaGetDeviceProperties(&properties, device);
				//taken from RQ's source code (RAN.cpp)
				if (max_multiprocessors < properties.multiProcessorCount) {
					max_multiprocessors = properties.multiProcessorCount;
					fastest_device = device;
				}
			}
			cout << "::: using CUDA device " << fastest_device << endl;
			cudaSetDevice(fastest_device);
		}else{
			cout << "no CUDA device available... aborting" << endl;
			exit(-1);
		}
	}

	//! Load SVM data (SVs and bias) from a file
	//! \param model_filename The filename where to read data from
	//! \param n_sv The number of Support Vectors in the file
	//! \param ndims The number of features in the file
	//! \param h_b The hyperplane's offset
	//! \param model A HostMatrix containing the model (features and alphas)
	//! \return 0 if successfully loaded the data, -1 otherwise
	int readModel(char * model_filename, int n_sv, int &ndims,
		cudafloat &h_b, GPUMLib::HostMatrix<cudafloat> &model) {
			cout << "loading model from file..." << endl;
			ifstream model_file(model_filename);
			if (model_file.is_open()) {
				//first line tells the amount of SVs
				model_file >> n_sv;
				//second tells the amount of features
				model_file >> ndims;
				ndims = ndims - 2;
				//third the hyperplanes offset
				model_file >> h_b;
				//create the model
				model.ResizeWithoutPreservingData(n_sv, ndims);
				for (int row = 0; row < model.Rows(); row++) {
					for (int col = 0; col < model.Columns(); col++) {
						cudafloat val;
						model_file >> val;
						model(row, col) = val;
					}
				}
				model_file.close();
				printf("read model from file %s with %d SVs and %d features\n", model_filename, n_sv, ndims);
				return 0;
			} else {
				cout << "Err: Unable to open model file for reading." << endl;
				return -1;
			}
	}

	//! Helper function to return precision delta time for 3 counters since last call based upon host high performance counter.
	//! Retrieved from shrUtils.h by NVIDIA.
	//! \param iCounterID The counter to be used (0, 1 or 2)
	//! \return The elapsed time since last call for the given counter
	double shrDeltaT(int iCounterID) {
		// local var for computation of microseconds since last call
		double DeltaT = -1.0;

#ifdef _WIN32 // Windows version of precision host timer
		// Variables that need to retain state between calls
		static LARGE_INTEGER liOldCount0 = { { 0, 0 } };
		static LARGE_INTEGER liOldCount1 = { { 0, 0 } };
		static LARGE_INTEGER liOldCount2 = { { 0, 0 } };

		// locals for new count, new freq and new time delta
		LARGE_INTEGER liNewCount, liFreq;
		if (QueryPerformanceFrequency(&liFreq)) {
			// Get new counter reading
			QueryPerformanceCounter(&liNewCount);

			// Update the requested timer
			switch (iCounterID) {
		case 0: {
			// Calculate time difference for timer 0.  (zero when called the first time)
			DeltaT = liOldCount0.LowPart ? (((double) liNewCount.QuadPart - (double) liOldCount0.QuadPart) / (double) liFreq.QuadPart) : 0.0;

			// Reset old count to new
			liOldCount0 = liNewCount;

			break;
				}
		case 1: {
			// Calculate time difference for timer 1.  (zero when called the first time)
			DeltaT = liOldCount1.LowPart ? (((double) liNewCount.QuadPart - (double) liOldCount1.QuadPart) / (double) liFreq.QuadPart) : 0.0;

			// Reset old count to new
			liOldCount1 = liNewCount;

			break;
				}
		case 2: {
			// Calculate time difference for timer 2.  (zero when called the first time)
			DeltaT = liOldCount2.LowPart ? (((double) liNewCount.QuadPart - (double) liOldCount2.QuadPart) / (double) liFreq.QuadPart) : 0.0;

			// Reset old count to new
			liOldCount2 = liNewCount;

			break;
				}
		default: {
			// Requested counter ID out of range
			return -9999.0;
				 }
			}

			// Returns time difference in seconds sunce the last call
			return DeltaT;
		} else {
			// No high resolution performance counter
			return -9999.0;
		}
#else
		// Linux version of precision host timer. See http://www.informit.com/articles/article.aspx?p=23618&seqNum=8
		static struct timeval _NewTime; // new wall clock time (struct representation in seconds and microseconds)
		static struct timeval _OldTime0;// old wall clock time 0(struct representation in seconds and microseconds)
		static struct timeval _OldTime1;// old wall clock time 1(struct representation in seconds and microseconds)
		static struct timeval _OldTime2;// old wall clock time 2(struct representation in seconds and microseconds)

		// Get new counter reading
		gettimeofday(&_NewTime, NULL);

		switch (iCounterID)
		{
		case 0:
			{
				// Calculate time difference for timer 0.  (zero when called the first time)
				DeltaT = ((double)_NewTime.tv_sec + 1.0e-6 * (double)_NewTime.tv_usec) - ((double)_OldTime0.tv_sec + 1.0e-6 * (double)_OldTime0.tv_usec);

				// Reset old time 0 to new
				_OldTime0.tv_sec = _NewTime.tv_sec;
				_OldTime0.tv_usec = _NewTime.tv_usec;

				break;
			}
		case 1:
			{
				// Calculate time difference for timer 1.  (zero when called the first time)
				DeltaT = ((double)_NewTime.tv_sec + 1.0e-6 * (double)_NewTime.tv_usec) - ((double)_OldTime1.tv_sec + 1.0e-6 * (double)_OldTime1.tv_usec);

				// Reset old time 1 to new
				_OldTime1.tv_sec = _NewTime.tv_sec;
				_OldTime1.tv_usec = _NewTime.tv_usec;

				break;
			}
		case 2:
			{
				// Calculate time difference for timer 2.  (zero when called the first time)
				DeltaT = ((double)_NewTime.tv_sec + 1.0e-6 * (double)_NewTime.tv_usec) - ((double)_OldTime2.tv_sec + 1.0e-6 * (double)_OldTime2.tv_usec);

				// Reset old time 2 to new
				_OldTime2.tv_sec = _NewTime.tv_sec;
				_OldTime2.tv_usec = _NewTime.tv_usec;

				break;
			}
		default:
			{
				// Requested counter ID out of range
				return -9999.0;
			}
		}

		// Returns time difference in seconds since the last call
		return DeltaT;
#endif
	}

	//! Main function to launch the SVM, either to train, classify or both
	int main(int argc, char **argv) {
		//disable stdout buffering
		setvbuf(stdout, NULL, _IONBF, 0);
		setvbuf(stderr, NULL, _IONBF, 0);
		bool train_model = false;
		bool classify_dataset = false;

		char * training_filename = NULL;
		char * testing_filename = NULL;
		char * model_filename = NULL;
		char * classification_results_filename = NULL;

		GPUMLib::svm_kernel_type kernel_type = SVM_KT_LINEAR;
		cudafloat * kernel_args = new cudafloat[4];
		kernel_args[0] = 1.0;
		kernel_args[1] = 1.0;
		kernel_args[2] = 1.0;
		kernel_args[3] = 1.0;

		cudafloat constant_c = CUDA_VALUE(1.0);
		cudafloat constant_c_negative = constant_c;
		cudafloat constant_c_positive = constant_c;

		cudafloat constant_epsilon = CUDA_VALUE(0.00001);
		cudafloat constant_tau = CUDA_VALUE(0.001);

		int amount_threads = MAX_THREADS_PER_BLOCK;

		bool arguments_error = false;
		int positive_class = 1;

		//read arguments and compile them
		GPUMLib::Settings settings(argc, argv);
		unsigned int aa = settings.getAmountArguments();
		//go through all arguments
		for (size_t i = 0; i < aa; i++) {
			Argument* a = settings.getArgument(i);
			//training file
			if (strcmp(a->argument, "-trainingset") == 0) {
				//cout << "extracting training file" << endl;
				if (a->value != NULL) {
					training_filename = a->value;
				} else {
					cout << "no training file was given" << endl;
					arguments_error = true;
				}
			}
			//classifying file
			else if (strcmp(a->argument, "-testingset") == 0) {
				//cout << "extracting testing file" << endl;
				if (a->value != NULL) {
					testing_filename = a->value;
				} else {
					cout << "no testing file was given" << endl;
					arguments_error = true;
				}
			}
			//classification results file
			else if (strcmp(a->argument, "-cr") == 0) {
				//cout << "extracting classification results file" << endl;
				if (a->value != NULL) {
					classification_results_filename = a->value;
				} else {
					cout << "no classification results file was given" << endl;
					arguments_error = true;
				}
			}
			//model file
			else if (strcmp(a->argument, "-model") == 0) {
				//cout << "extracting model file" << endl;
				if (a->value != NULL) {
					model_filename = a->value;
				} else {
					cout << "no model file given" << endl;
					arguments_error = true;
				}
			}
			//train?
			else if (strcmp(a->argument, "-train") == 0) {
				//cout << "user wants to train model" << endl;
				train_model = true;
			}
			//classify?
			else if (strcmp(a->argument, "-classify") == 0) {
				//cout << "user wants to classify dataset" << endl;
				classify_dataset = true;
			}
			//kernel type
			else if (strcmp(a->argument, "-k") == 0) {
				//cout << "extracting kernel type" << endl;
				if (a->value != NULL) {
					if (strcmp(a->value, "lin") == 0) {
						kernel_type = SVM_KT_LINEAR;
					} else if (strcmp(a->value, "pol") == 0) {
						kernel_type = SVM_KT_POLYNOMIAL;
					} else if (strcmp(a->value, "rbf") == 0) {
						kernel_type = SVM_KT_RBF;
					} else if (strcmp(a->value, "sig") == 0) {
						kernel_type = SVM_KT_SIGMOID;
					} else if (strcmp(a->value, "ukf") == 0) {
						kernel_type = SVM_KT_UKF;
					} else {
						cout << "unknown kernel type: " << a->value << endl;
						arguments_error = true;
					}
				} else {
					cout << "no kernel type was given" << endl;
					arguments_error = true;
				}
			}
			//kernel arguments
			//a
			else if (strcmp(a->argument, "-a") == 0) {
				//cout << "extracting argument <a>" << endl;
				if (a->value != NULL) {
					kernel_args[0] = (cudafloat) atof(a->value);
				} else {
					cout << "no argument <a> was given" << endl;
					arguments_error = true;
				}
			}
			//b
			else if (strcmp(a->argument, "-b") == 0) {
				//cout << "extracting argument <b>" << endl;
				if (a->value != NULL) {
					kernel_args[1] = (cudafloat) atof(a->value);
				} else {
					cout << "no argument <b> was given" << endl;
					arguments_error = true;
				}
			}
			//c
			else if (strcmp(a->argument, "-c") == 0) {
				//cout << "extracting argument <c>" << endl;
				if (a->value != NULL) {
					kernel_args[2] = (cudafloat) atof(a->value);
				} else {
					cout << "no argument <c> was given" << endl;
					arguments_error = true;
				}
			}
			//penalization constant
			else if (strcmp(a->argument, "-C") == 0) {
				//cout << "extracting penalization constant C" << endl;
				if (a->value != NULL) {
					constant_c_negative = (cudafloat) atof(a->value);
					constant_c_positive = (cudafloat) atof(a->value);
				} else {
					cout << "no penalization constant was given" << endl;
					arguments_error = true;
				}
			}
			//negative penalization constant
			else if (strcmp(a->argument, "-Cn") == 0) {
				//cout << "extracting penalization constant C" << endl;
				if (a->value != NULL) {
					constant_c_negative = (cudafloat) atof(a->value);
				} else {
					cout << "no negative penalization constant was given" << endl;
					arguments_error = true;
				}
			}
			//positive penalization constant
			else if (strcmp(a->argument, "-Cp") == 0) {
				//cout << "extracting penalization constant C" << endl;
				if (a->value != NULL) {
					constant_c_positive = (cudafloat) atof(a->value);
				} else {
					cout << "no positive penalization constant was given" << endl;
					arguments_error = true;
				}
			}
			//optimality conditions tolerance
			else if (strcmp(a->argument, "-eps") == 0) {
				//cout << "extracting optimality conditions tolerance" << endl;
				if (a->value != NULL) {
					constant_epsilon = (cudafloat) atof(a->value);
				} else {
					cout << "no optimality conditions tolerance was given" << endl;
					arguments_error = true;
				}
			}
			//optimality gap size
			else if (strcmp(a->argument, "-tau") == 0) {
				//cout << "extracting optimality gap size" << endl;
				if (a->value != NULL) {
					constant_tau = (cudafloat) atof(a->value);
				} else {
					cout << "no optimality gap size was given" << endl;
					arguments_error = true;
				}
			}
			//amount of threads
			else if (strcmp(a->argument, "-threads") == 0) {
				//cout << "extracting amount of threads" << endl;
				if (a->value != NULL) {
					amount_threads = atoi(a->value);
				} else {
					cout << "no amount of threads was given" << endl;
					arguments_error = true;
				}
			}
			//positive class
			else if	(strcmp(a->argument, "-positive") == 0) {
				if (a->value != NULL) {
					positive_class = atoi(a->value);
				} else {
					cout << "positive label was not given" << endl;
					arguments_error = true;
				}
			}
		}

		//for training we require the training dataset... duh
		if (train_model) {
			if (training_filename == NULL) {
				cout << "Error: no training dataset was given - Aborting." << endl;
				arguments_error = true;
			} else {
				//cout << "training dataset is " << training_filename << endl;
			}
		}

		//for classifying we require both the training and testing datasets
		if (classify_dataset) {
			//if in this execution the model is not trained, it must be read from somewhere...
			if (train_model == false && model_filename == NULL) {
				cout << "Error: no model file was given." << endl;
				return -1;
			} else {
				//cout << "model file is " << model_filename << endl;
			}
			if (testing_filename == NULL) {
				cout << "Error: no testing dataset was given." << endl;
				return -1;
			} else {
				//cout << "testing dataset is " << model_filename << endl;
			}
		}

		if (classify_dataset == false && train_model == false) {
			cout << "Error: the program was not instructed to train nor to classify." << endl;
			arguments_error = true;
		}

		if (arguments_error) {
			cout << "Error: invalid arguments." << endl;
			cout << "----------------------------------------------------------" << endl;
			cout << "The arguments are the following:" << endl;
			cout << "" << endl;

			cout << "to train using the training samples" << endl;
			cout << "\t -train" << endl;
			cout << "" << endl;

			cout << "to classify using the trained svm model" << endl;
			cout << "\t -classify" << endl;
			cout << "" << endl;

			cout << "file with the training set (filename) - required:" << endl;
			cout << "\t -trainingset <training file>" << endl;
			cout << "" << endl;

			cout << "file with the testing set (filename) - required:" << endl;
			cout << "\t -testingset <training file>" << endl;
			cout << "" << endl;

			cout << "file where to store the trained svm model (filename):" << endl;
			cout << "\t -model <output file>" << endl;
			cout << "" << endl;

			cout << "file where to store the classification results (filename):" << endl;
			cout << "\t -cr <output file>" << endl;
			cout << "" << endl;

			cout << "which kernel to use (text):" << endl;
			cout << "\t -k <type>" << endl;
			cout << "\t where <type> can be one of the following:" << endl;
			cout << "\t\t lin - for the linear kernel: K(x1,x2) = x1.x2" << endl;
			cout << "\t\t pol - for the polynomial kernel: K(x1,x2) = a*(x1.x2+b)^c" << endl;
			cout << "\t\t rbf - for the gaussian kernel: K(x1,x2) = e^(-a*||x1-x2||^2)" << endl;
			cout << "\t\t sig - for the sigmoid kernel: K(x1,x2) = tanh(a*(x1.x2)+b)" << endl;
			cout << "\t\t ukf - for the universal function kernel: K(x1,x2) = a*(||x1-x2||^2+b^2)^-c" << endl;
			cout << "\t being x1.x2 the dot product between vectors x1 and x2" << endl;
			cout << "" << endl;

			cout << "kernel arguments (decimal number):" << endl;
			cout << "\t -a <value>" << endl;
			cout << "\t -b <value>" << endl;
			cout << "\t -c <value>" << endl;
			cout << "" << endl;

			cout << "penalization constant C (decimal number):" << endl;
			cout << "\t -C <value>" << endl;
			cout << "" << endl;

			cout << "optimality conditions tolerance, Epsilon, which allows some numerical uncertainty on the heuristics (decimal number):" << endl;
			cout << "\t -eps <value>" << endl;
			cout << "" << endl;

			cout << "optimality gap size, Tau, which regulates the training convergence (decimal number):" << endl;
			cout << "\t -tau <value>" << endl;
			cout << "" << endl;

			cout << "amount of threads to use in trainer and classifier (integer, 0 = automatic):" << endl;
			cout << "\t -threads <value>" << endl;
			cout << "" << endl;

			cout << "ABORTING." << endl;
			return -1;
		}

		switch (kernel_type) {
	case SVM_KT_RBF:
		if (DEBUG)
			cout << "using RBF kernel with gamma = " << kernel_args[0] << endl;
		break;
	case SVM_KT_LINEAR:
		if (DEBUG)
			cout << "using linear kernel" << endl;
		break;
	case SVM_KT_POLYNOMIAL:
		if (DEBUG)
			cout << "using polynomial kernel" << endl;
		break;
	case SVM_KT_SIGMOID:
		if (DEBUG)
			cout << "using sigmoid kernel" << endl;
		break;
	case SVM_KT_UKF:
		if (DEBUG)
			cout << "using universal kernel function with L = " << kernel_args[0] << " b (sigma) = " << kernel_args[1] << " and c (alpha) = " << kernel_args[2]
		<< endl;
		break;
		}

		if (constant_c_negative <= 0 || constant_c_positive <= 0) {
			cout << "Error: invalid value for C" << endl;
			return -1;
		}
		if (DEBUG)
			cout << "C negative = " << constant_c_negative << " C positive = " << constant_c_positive << endl;
		if (DEBUG)
			cout << "epsilon = " << constant_epsilon << endl;
		if (constant_tau <= 0) {
			cout << "Error: invalid value for epsilon" << endl;
			return -1;
		}
		if (DEBUG)
			cout << "tau = " << constant_tau << endl;

		// read training dataset file
		// read training dataset file
		// read training dataset file


		//create a matrix to hold the model
		//structure: alpha_i | class_i | attribute_0 ... attribute_n-1
		GPUMLib::HostMatrix<cudafloat> h_model(1, 1, GPUMLib::ColumnMajor);
		int n_sv = -1;
		int ndims = -1;
		cudafloat h_b = CUDA_VALUE(0.0);

		selectFastestDevice();

		//create a instance to manage the GPU SVM
		GPUMLib::SVM svm;

		//train model if requested
		if (train_model) {

			//build matrix for holding training data set
			cout << "reading training dataset file " << training_filename << endl;
			FILE *f_input = fopen(training_filename, "r");
			if (f_input == NULL) {
				cout << "error while reading training dataset file" << endl;
				return -1;
			}
			ndims = getNumberOfColumns(f_input) - 1;
			int training_dataset_size = getNumberOfSamples(f_input);
			//cout << "allocating storage for training dataset:" << training_filename << endl;

			//create the storage in hosts memory for the dataset
			GPUMLib::HostMatrix<cudafloat> h_samples(training_dataset_size, ndims, GPUMLib::ColumnMajor);
			GPUMLib::HostArray<int> h_classes(training_dataset_size);

			// create data structures for storing alphas
			GPUMLib::HostArray<cudafloat> h_alphas(training_dataset_size);

			//read the dataset
			readDataSet(f_input, h_samples, h_classes, ndims + 1, positive_class);
			fclose(f_input);

			shrDeltaT(1);
			shrDeltaT(1);
			double t0 = shrDeltaT(1);

			svm.train(h_samples, h_classes, constant_c_negative, constant_c_positive,
				constant_epsilon, constant_tau, kernel_type, kernel_args, amount_threads, h_alphas,
				//training_dataset_size,
				n_sv, h_model,
				//ndims, 
				h_b);
			double t1 = shrDeltaT(1);
			printf("training took %f s\n", t1 - t0);

			//if requested save model to a file
			if (model_filename != NULL) {
				saveModel(model_filename, h_model, h_b);
			}
		}

		if (classify_dataset) {
			puts("------------------");

			// if in this call the model hasn't been created, load it
			if (!train_model) {
				if (readModel(model_filename, n_sv, ndims, h_b, h_model) != 0) {
					cout << "error while reading model" << endl;
					return -1;
				}
			}
			printf("using model with %d SVs and %d features\n", n_sv, ndims);

			// read testing dataset file
			// read testing dataset file
			// read testing dataset file

			//build matrix for holding testing data set
			cout << "reading testing dataset file " << testing_filename << endl;
			FILE *f_input_test = fopen(testing_filename, "r");
			if (f_input_test == NULL) {
				cout << "error while reading testing dataset file" << endl;
				return -1;
			}
			int testing_dataset_size = getNumberOfSamples(f_input_test);
			GPUMLib::HostMatrix<cudafloat> h_testing_samples(testing_dataset_size, ndims, GPUMLib::ColumnMajor);
			GPUMLib::HostArray<int> h_testing_classes(testing_dataset_size);

			//read the dataset
			readDataSet(f_input_test, h_testing_samples, h_testing_classes, ndims + 1, positive_class);
			fclose(f_input_test);

			// start classifying phase
			shrDeltaT(1);
			shrDeltaT(1);
			double t0 = shrDeltaT(1);

			GPUMLib::HostArray<int> h_testing_results(testing_dataset_size);
			svm.classify(h_model, h_testing_samples, kernel_args, amount_threads, kernel_type, n_sv, h_b, ndims, h_testing_results);

			double t1 = shrDeltaT(1);
			printf("classification took %f s\n", t1 - t0);
			// 		for (int i=0;i<h_testing_results.Length();i++){
			// 			printf("target\t%d\tpredicted\t%d\n", h_testing_classes[i], h_testing_results[i]);
			// 		}

			showClassificationMetrics(h_testing_classes, h_testing_results, testing_dataset_size);

			//if requested save results to a file
			if (classification_results_filename != NULL) {
				if (DEBUG)
					cout << "saving classification results to file " << classification_results_filename << endl;

				char * WRITE_BUF = new char[BUFFERING_BLOCK_SIZE];
				FILE *model_file;
				model_file = fopen(classification_results_filename, "w");
				if (model_file) {
					//buffer process
					setvbuf(model_file, WRITE_BUF, _IOFBF, BUFFERING_BLOCK_SIZE);
					//give the amount of samples
					fprintf(model_file, "#%d\n", testing_dataset_size);
					//first line gives a comment
					fprintf(model_file, "target,predicted\n");
					for (int i = 0; i < testing_dataset_size; i++) {
						fprintf(model_file, "%d,%d\n", h_testing_classes[i], h_testing_results[i]);
					}
					fclose(model_file);
				} else
					cout << "Err: Unable to open classification results file for write." << endl;
				delete WRITE_BUF;
			}
		}

		delete kernel_args;

		if (DEBUG)
			cout << "exiting..." << endl;
		return 0;
	}
} //namespace

int main(int argc, char **argv){
	GPUMLib::main(argc, argv);
}
