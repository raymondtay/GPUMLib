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

#include "ResourceAllocatingNetwork.h"

namespace GPUMLib {

ResourceAllocatingNetwork::ResourceAllocatingNetwork(float scale_of_interest_max, float desired_accuracy, float overlap_factor, int Rows, int Columns, int NumClasses){


	this->scale_of_interest_max = scale_of_interest_max;
	// this->scale_of_interest_min = scale_of_interest_min;
	this->scale_of_interest = scale_of_interest_max;

	//this->decay = decay;
	this->desired_accuracy = desired_accuracy;

	//this->alpha = alpha;
	this->overlap_factor = overlap_factor;

	this->MaxWidth = 1;

	this->start = clock();

	this->times[0] = 0;
	this->times[1] = 0;
	this->times[2] = 0;
	this->times[3] = 0;


	this->Columns = Columns;
	this->NumCenters = 0;
	this->NumMemoryItems = 1;
	this->NumClasses = NumClasses;


	this->dCenters = DeviceMatrix<float>(Rows,Columns);
	this->dWeights = DeviceMatrix<float>(Rows,NumClasses);
	this->dWidths = DeviceArray<float>(Rows);

	this->dIMemory = DeviceMatrix<float>(Rows,Columns);  
	this->dTMemory_2 = DeviceMatrix<float>(Rows,NumClasses);

	this->CounterCenters = (int*)malloc(sizeof(int)*Rows);
	memset(this->CounterCenters,0,sizeof(int)*Rows);


	cudaMalloc((void **) &(this->derror),sizeof(float)); 
	cudaMalloc((void **) &(this->ddistance),sizeof(float));


	cudaMalloc((void **) &result, NumClasses * sizeof(float));


	this->Distances = DeviceArray<float>(Rows);
	this->error_arr = DeviceArray<float>(NumClasses);


}

ResourceAllocatingNetwork::~ResourceAllocatingNetwork(){
	free(this->CounterCenters);
	cudaFree(this->derror);
	cudaFree(this->ddistance);
}


float ResourceAllocatingNetwork::FindMaxWidth(DeviceMatrix<float> &device_X, DeviceMatrix<float> &Y){

	DeviceMatrix<float> device_output2(device_X.Rows(),device_X.Rows());

	KernelEuclidianDistance(device_output2.Pointer(), device_output2.Rows(), device_output2.Columns(), device_X.Pointer(), device_X.Columns(), device_X.Pointer(), device_X.Columns());

	DeviceArray<float> dMin(device_output2.Rows());
	DeviceArray<int> dIdx(device_output2.Rows());

	//Min
	FindMin(device_output2.Pointer(),device_output2.Rows(), device_output2.Columns(),dMin.Pointer(),dIdx.Pointer(),Y.Pointer());

	HostArray<float> hMin(dMin);
	//HostArray<int> hIdx(dIdx);

	std::sort(hMin.Pointer(), hMin.Pointer() + hMin.Length());

	//Median
	MaxWidth = hMin[(int)(device_output2.Rows()/2)];

	return MaxWidth;
}

void ResourceAllocatingNetwork::FindNearestCenter(cudafloat *Sample, int Length,float *value, int* index,cudafloat* output){
	KernelFindNearestCenter(output, NumCenters, Sample, dCenters.Pointer(), dCenters.Columns(),value);
}

float* ResourceAllocatingNetwork::CalculateNetworkActivation(cudafloat *Sample, int Length){

	DeviceMatrix<float> output(NumCenters,NumClasses);

	KernelCalculateNetworkActivation(output.Pointer(),Sample,Length,dCenters.Pointer(),NumCenters,dWeights.Pointer(),NumClasses,dWidths.Pointer(),overlap_factor);
	KernelSumActivations(output.Pointer(),NumClasses,NumCenters);

	cudaMemcpy(result,output.Pointer(),sizeof(float)*NumClasses,cudaMemcpyDeviceToDevice);  

	return result;
}

void ResourceAllocatingNetwork::AddCenter(cudafloat *Sample,int Length, float* Width, float* Weight){

	cudaMemcpy(&(dCenters.Pointer()[NumCenters*dCenters.Columns()]),Sample,sizeof(float)*Length,cudaMemcpyDeviceToDevice);
	cudaMemcpy(&(dWeights.Pointer()[NumCenters*dWeights.Columns()]),Weight,sizeof(float)*NumClasses,cudaMemcpyDeviceToDevice);   
	cudaMemcpy(&(dWidths.Pointer()[NumCenters]),Width,sizeof(float),cudaMemcpyDeviceToDevice);   

	// if(Width < MaxWidth)

	// else 
	//      cudaMemcpy(&(dWidths.Pointer()[NumCenters]),&MaxWidth,sizeof(float),cudaMemcpyHostToDevice);

	CounterCenters[NumCenters] = 1;

	NumCenters = NumCenters + 1;

}


void ResourceAllocatingNetwork::AddMemory(cudafloat *Sample,int Length, float* Target){

	cudaMemcpy(&(dIMemory.Pointer()[NumMemoryItems*dIMemory.Columns()]),Sample,sizeof(float)*Length,cudaMemcpyDeviceToDevice);
	cudaMemcpy(&(dTMemory_2.Pointer()[NumMemoryItems*dTMemory_2.Columns()]),Target,sizeof(float)*NumClasses,cudaMemcpyDeviceToDevice);

	NumMemoryItems = NumMemoryItems + 1;

}

void ResourceAllocatingNetwork::UpdateWeights(cudafloat *Sample,int Length, float* Target){

	cudaMemcpy(dIMemory.Pointer(),Sample,sizeof(float)*Length,cudaMemcpyDeviceToDevice);
	cudaMemcpy(dTMemory_2.Pointer(),Target,sizeof(float)*NumClasses,cudaMemcpyDeviceToDevice);

	DeviceMatrix<float> device_output2(NumMemoryItems,NumCenters,ColumnMajor);

	KernelActivationMatrix(device_output2.Pointer(), device_output2.Rows(), device_output2.Columns(), 
		dIMemory.Pointer(), dIMemory.Columns(),
		dCenters.Pointer(), dCenters.Columns(),
		dWidths.Pointer(),overlap_factor);

	UTILS::pseudoinverse2(device_output2);

	matmul(dWeights.Pointer(), device_output2.Pointer(), dTMemory_2.Pointer(),device_output2.Rows(),dTMemory_2.Columns(), dTMemory_2.Columns(), device_output2.Columns());
}

void ResourceAllocatingNetwork::Train(cudafloat *Sample,int Length,float Target, cudafloat* dTargetArr){

	if(NumCenters == 0){

		float* dMaxWidth; cudaMalloc((void **) &dMaxWidth,sizeof(float));
		cudaMemcpy(dMaxWidth,&MaxWidth,sizeof(float),cudaMemcpyHostToDevice);

		AddCenter(Sample,Length,dMaxWidth,dTargetArr);
		AddMemory(Sample,Length,dTargetArr);

		cudaFree(dMaxWidth);

		return;
	}

	//float distance; 
	int index; 	
	//float error;
	// UTILS::checkGpuMem("3");

	CalculateNetworkActivation(Sample,Length);



	FindNearestCenter(Sample,Length,ddistance,&index,Distances.Pointer());
	var_distance.UpdateValue(ddistance);

	KernelCalculateError(result,dTargetArr,error_arr.Pointer(),NumClasses,derror);
	var_error.UpdateValue(derror);


	//std::cout << Target << " " << var_error.Value() << " " << var_distance.Value() << std::endl;


	if(var_error.Value() > desired_accuracy && var_distance.Value() > scale_of_interest){

		AddCenter(Sample,Length,ddistance,error_arr.Pointer());
		KernelUpdateWidths(dWidths.Pointer(),Distances.Pointer(),NumCenters-1);
		AddMemory(Sample,Length,dTargetArr);

	}else{

		UpdateWeights(Sample,Length,dTargetArr);

		//run input through the network again (step 5 of IncrementaLearningofFeatureSpace-Seiichi)  
		CalculateNetworkActivation(Sample,Length);

		FindNearestCenter(Sample,Length,ddistance,&index,Distances.Pointer());
		KernelCalculateError(result,dTargetArr,error_arr.Pointer(),NumClasses,derror);

		var_error.UpdateValue(derror);

		//std::cout << "--> error " << var_error.Value() << std::endl;

		if(var_error.Value() > desired_accuracy){

			AddCenter(Sample,Length,ddistance,error_arr.Pointer());  
			KernelUpdateWidths(dWidths.Pointer(),Distances.Pointer(),NumCenters-1);           
			AddMemory(Sample,Length,dTargetArr);   

		}


	}

}

}