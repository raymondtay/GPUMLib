/*
	Ricardo Quintas is an MSc Student at the University of Coimbra, Portugal
    Copyright (C) 2009, 2010 Ricardo Quintas

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

#include <cuda_runtime.h>

#include <iostream> 
#include <sstream>
#include <stdio.h>
#include <string.h>

#include "../../memory/HostMatrix.h"
#include "../Dataset/Dataset.h"
#include "../../RBF/RadialBasisFunction.h"
#include "../../RBF/utils.h"
#include "../common/CudaInit.h"

#define round(x) (x<0?ceil((x)-0.5):floor((x)+0.5))

#define FILE_NAME "satellite_n.data"
#define NETWORK_SIZE 150
#define RNEIGHBOURS 3
#define KFOLDS 5
#define SCALING_FACTOR 1

bool InitCUDA(void)
{
	int count = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	CudaDevice device;
	if (!device.SupportsCuda()) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}

	cudaEvent_t wakeGPU;
	cudaEventCreate( &wakeGPU) ;

	printf("CUDA initialized.\n");
	return true;
}


/*******/
void writeHeader(HostMatrix<float> &Input, int number_classes, int seed){

	cout << "\n=== Run information ===\n\n";

	cout << std::left << setw(50) << "Random Seed" << std::left << setw(20) << seed << endl;
	cout << std::left << setw(50) << "Number of Basis Functions" << std::left << setw(20) << NETWORK_SIZE << endl;
	cout << std::left << setw(50) << "Number of Neighbours for Width Estimation" << std::left << setw(20) << RNEIGHBOURS << endl;
	cout << std::left << setw(50) << "Number of Folds" << std::left << setw(20) << KFOLDS << endl;
	cout << std::left << setw(50) << "Number of Attributes" << std::left << setw(20) << Input.Columns() << endl;
	cout << std::left << setw(50) << "Number of Classes" << std::left << setw(20) << number_classes << endl;
	cout << std::left  << setw(50) << "Number of Instances" << std::left << setw(20) << Input.Rows() << endl;

}

void writeFooter(float center_time, float width_time, float weight_time, float scaling_time, unsigned int training_time, unsigned int testing_time, unsigned int time_total){

	std::cout << "\nCenter Selection: " <<  (center_time)/1000 
		<< "\nWidth Selection: " << (width_time)/1000 
		<< "\nWeight Selection: " <<  (weight_time)/1000 << "\n";

	std::cout << endl << "Total Time: "<< time_total / 1000 << std::endl;

}

int main(int argc, char* argv[]){

	//float beta;
	//int row,col;

	string file_name = FILE_NAME;

	HostMatrix<float> X;
	HostMatrix<float> X_test; 
	HostMatrix<float> Y;
	HostMatrix<float> Y_test;

	HostMatrix<float> Input;
	HostMatrix<float> Target;

	std::map<string,int> Classes;
	std::map<int,string> ClassesLookup;

	readFile(file_name,Input,Target,Classes,ClassesLookup);

	int kfold = 1;
	int correct_instances = 0;
	int incorrect_instances = 0;
	int total_instances = 0;

	int **confusionMatrix;

	confusionMatrix = (int**) malloc(sizeof(int*)*Classes.size());

	for(int i = 0; i < (int)Classes.size(); i++){
		confusionMatrix[i] = (int*) malloc(sizeof(int)*Classes.size());
		memset(confusionMatrix[i],0,sizeof(int)*Classes.size());
	}


	float Pet_mean = 0;
	float Ped_mean = 0;

	unsigned int seed = (unsigned)time(0);

	/***************RUN INFORMATION*************/

	writeHeader(Input,Classes.size(),seed);

	/*******************************************/


	if(!InitCUDA()) {
		return 1;
	}

	culaStatus status;
	status = culaInitialize();


	std::cout << "Starting " << std::endl;


	float center_time = 0;
	float width_time = 0;
	float weight_time = 0;
	float scaling_time = 0;

	unsigned int time_total = 0;
	unsigned int testing_time = 0;
	unsigned int training_time = 0;

	clock_t initialTimeTotal = clock();

	do{	
		X = crossvalidationTrain(Input,KFOLDS,kfold);
		X_test = crossvalidationTest(Input,KFOLDS,kfold);
		Y = crossvalidationTrain(Target,KFOLDS,kfold);
		Y_test = crossvalidationTest(Target,KFOLDS,kfold);

		HostMatrix<float> Weights;
		HostMatrix<float> Centers;

		/*Train Network*/

		clock_t initialTime = clock();
		RadialBasisFunction RBF(NETWORK_SIZE,RNEIGHBOURS,SCALING_FACTOR,Classes.size());
		RBF.SetSeed(seed);
		RBF.Train(X,Y);
		training_time = (clock() - initialTime);

		center_time += RBF.times[0];
		width_time += RBF.times[1];
		weight_time += RBF.times[2];
		scaling_time += RBF.times[3];

		/*Test Network*/

		initialTime = clock();
		std::cout << "Testing" << std::endl;
		HostMatrix<float> out_test;


		out_test = RBF.Test(X_test);

		for(int i = 0; i< X_test.Rows();i++){

			float max = 0;
			float out_class = 0;
			for(int j = 0; j < (int) Classes.size(); j++){
				if(out_test(i,j) > max){
					out_class = (float)j;
					max = out_test(i,j);
				}
			}

			out_test(i,0) = out_class+1;

		}

		for (int i = 0; i < out_test.Rows(); i++)
		{

			out_test(i,0) = (float)round(out_test(i,0));     

			if(out_test(i,0) <= 0) out_test(i,0) = 1;

			if(out_test(i,0) > Classes.size()) out_test(i,0) = (float)Classes.size();

			std::cout << Y_test(i,0) << " " << out_test(i,0) << std::endl;
		}


		correct_instances += out_test.Rows() - error_calc(Y_test,out_test);
		incorrect_instances += error_calc(Y_test,out_test);
		total_instances += out_test.Rows();

		/*Add values to Confusion Matrix*/
		for(int i = 0; i < Y_test.Rows(); i++){
			confusionMatrix[((int)Y_test(i,0))-1][((int)out_test(i,0))-1] = confusionMatrix[((int)Y_test(i,0))-1][((int)out_test(i,0))-1] + 1;
		}

		testing_time = (clock() - initialTime);

		/*Increment fold number, for use in crossvalidation*/
		kfold++;
	}while(kfold <= KFOLDS);

	time_total  = (clock() - initialTimeTotal);

	/*****************MEASURES****************/

	measures(correct_instances,total_instances,incorrect_instances,confusionMatrix,Classes,ClassesLookup);

	writeFooter(center_time,width_time,weight_time,scaling_time,training_time,testing_time,time_total);

	culaShutdown();
	cudaThreadExit();

	return 0;
}


