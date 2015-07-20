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

#ifndef ResourceAllocatingNetwork_h
#define ResourceAllocatingNetwork_h

#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../common/CudaDefinitions.h"
#include "../common/CudaStreams.h"
#include "../memory/HostMatrix.h"
#include "../memory/DeviceMatrix.h"
#include "../memory/DeviceAccessibleVariable.h"
#include "../RAN/rankernels.h"

#include "../RAN/utils.h"

#include <ctime>

#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))
#define round(x) (x<0?ceil((x)-0.5):floor((x)+0.5))

using namespace std;

namespace GPUMLib {

//! \addtogroup ran Resource Allocating Network class
//! @{

//! Represents a resource allocating network with long term memory that can be trained using a CUDA implementation of the algorithm.
class ResourceAllocatingNetwork{

private:

	float scale_of_interest;
	float scale_of_interest_max;
	float scale_of_interest_min;

	float decay;

	float alpha;
	float desired_accuracy;


	float overlap_factor;

	HostMatrix<float> Input;
	HostMatrix<float> Target;

	DeviceMatrix<float> dCenters;		
	DeviceMatrix<float> dIMemory;
	DeviceArray<float> dTMemory;
	DeviceMatrix<float> dWeights;
	DeviceArray<float> dWidths;


	HostMatrix<float> Centers;
	HostArray<float> Weights;
	HostArray<float> Widths;

	HostMatrix<float> Weights_2;
	HostMatrix<float> TMemory_2;
	DeviceMatrix<float> dTMemory_2;

	HostMatrix<float> IMemory;
	HostArray<float> TMemory;

	int NumClasses;

	int NumCenters;
	int NumMemoryItems;
	int Columns;
	float MaxWidth;

	int* CounterCenters;

	float* derror; 
	float* ddistance;
	float* herror;
	float* hdistance;

	float* result;

	DeviceAccessibleVariable<float> var_error;
	DeviceAccessibleVariable<float> var_distance;

	CudaStream stream1;

	DeviceArray<float> Distances;
	DeviceArray<float> error_arr;

	void FindNearestCenter(cudafloat* Sample, int Length,float *value, int* index,cudafloat *output);
	void AddCenter(cudafloat* Sample, int Length, float* Width, float* Weight);
	void AddMemory(cudafloat* Sample, int Length, float* Target);
	void UpdateWeights(cudafloat* Sample, int Length, float* Target);

public:

	unsigned int start;
	unsigned int times[4];

	float center_time;
	float width_time;
	float weight_time;
	float scaling_time;

	//! Constructs a resource allocating network with long term memory.
	//! \param scale_of_interest_max Cut off value for distance.
	//! \param desired_accuracy Cut off value for the accuracy.
	//! \param overlap_factor Scaling factor to apply to the width of the neuron.
	//! \param Rows Number of rows in the training data, used to reserve memory.
	//! \param Columns Number of features.
	//! \param NumClasses Total number of classes, if it is a regression problem use the value 1 (one).
	ResourceAllocatingNetwork(float scale_of_interest_max, float desired_accuracy, float overlap_factor, int Rows, int Columns, int NumClasses);
	~ResourceAllocatingNetwork();

	//! Trains the network with the given sample.
	//! \param Sample training array.
	//! \param Length of the sample.  
	//! \param Target desired output.
	//! \param dTargetArr desired activation in each output neuron.
	void Train(cudafloat* Sample, int Length,float Target, float* dTargetArr);

	//! Calculates the maximum width for the neurons in the hidden layer.
	//! \param X matrix with the training data.
	//! \param Y matrix with the targets.
	//! \return Value of the maximum width.
	float FindMaxWidth(DeviceMatrix<float> &X,DeviceMatrix<float> &Y);  

	//! Calculates the output of the network for the given sample.
	//! \param Sample array.
	//! \param Length of the sample (number of features).
	//! \return Network activation value.
	float* CalculateNetworkActivation(cudafloat* Sample, int Length);

	int GetNumCenters(){
		return NumCenters;
	}
};

//! \example RAN.cpp 
//! Example of the CUDA Resource Allocating Network algorithm usage.

//! @}

}

#endif