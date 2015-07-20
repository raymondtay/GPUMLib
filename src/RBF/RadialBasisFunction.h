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

#ifndef RadialBasisFunction_h
#define RadialBasisFunction_h

#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>

#include "../common/CudaDefinitions.h"
#include "../memory/HostMatrix.h"
#include "../memory/DeviceMatrix.h"

#include "../RBF/utils.h"
#include "../RBF/rbfkernels.h"

#include "../KMeans/KMeans.h"

#include <ctime>

#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))
#define imax(X, Y)  ((X) > (Y) ? (X) : (Y))
#define round(x) (x<0?ceil((x)-0.5):floor((x)+0.5))

using namespace std;

namespace GPUMLib {

//! \addtogroup rbf Radial Basis Function Network class
//! @{

//! Represents a radial basis function network that can be trained using the CUDA implementation of the Radial Basis Function algorithm.
class RadialBasisFunction{

private:

	int network_size;
	int number_neighbours;

	//HostMatrix<float> Weights;
	//HostMatrix<float> Centers;

	DeviceMatrix<float> dWeights;
	DeviceMatrix<float> dCenters;

	DeviceArray<float> device_c_width;

	//float *c_width;
	float scaling_factor;

	int NumClasses;

	unsigned int seed;

public:



	unsigned int start;
	unsigned int times[4];

	//! Constructs a radial basis function network that is trained using the a CUDA implementation of the algorithm.
	//! \param network_size Number of neurons for the hidden layer.
	//! \param number_neighbours Number of neighbours to use in the width estimation.
	//! \param scaling_factor Scaling factor to apply to the width of the neuron.
	//! \param NumClasses Total number of classes, if it is a regression problem use the value 1 (one).
	RadialBasisFunction(int network_size, int number_neighbours, float scaling_factor, int NumClasses);

	~RadialBasisFunction();

	//! Trains the network.
	//! \param Input matrix.
	//! \param Target target matrix.
	void Train(HostMatrix<float> &Input,HostMatrix<float> &Target);


	//! Test the network against the input data.
	//! \param Input matrix to test against.
	//! \return Solution matrix.
	HostMatrix<float> Test(HostMatrix<float> &Input);

	//! Test the network against the input data.
	//! \param Input matrix to test against.
	//! \return Solution matrix.
	DeviceMatrix<float> Test(DeviceMatrix<float> &Input);

	//! Test the network against the input data, using the given Centers and Weights.
	//! \param Input matrix to test against.
	//! \param Centers matrix with the centers to use.
	//! \param Weights matrix with the weights to use.
	//! \param Widths array with the widths to use.     
	//! \return Solution matrix.
	DeviceMatrix<float> Test(DeviceMatrix<float> &Input,DeviceMatrix<float> &Centers,DeviceMatrix<float> &Weights, DeviceArray<float> &Widths);

	//! Calculates the widths of the neurons, using the R closest neighbours.
	//! \param rneighbours number of neighbours to use.
	void AdjustWidths(int rneighbours);      

	//! Set scaling factor to be applied on the hidden layer neuron width.
	//! \param scaling_factor Scaling factor to use.
	void SetScalingFactor(float scaling_factor){
		this->scaling_factor = scaling_factor;
	}

	//! Get scaling factor applied on the hidden layer neuron width.
	//! \return Scaling factor used.
	float GetScalingFactor(){
		return scaling_factor;
	}

	void SetSeed(unsigned int seed){
		this->seed = seed;
	}

	unsigned int GetSeed(){
		return seed;
	}

	//! Get the weights of the network.
	//! \return Matrix with the weights of the network.
	DeviceMatrix<float> GetWeights(){
		return dWeights;
	}

	//! Get the hidden layer center values.
	//! \return Matrix with the centers of the network.
	DeviceMatrix<float> GetCenters(){
		return dCenters;
	}

	//! Get the widths of the hidden layer gaussian functions.
	//! \return Array with the widths applied in the hidden layer.
	DeviceArray<float> & GetWidths(){
		return device_c_width;
	}
};

//! \example RBF.cpp 
//! Example of the CUDA Radial Basis Function algorithm usage.

//! @}

}

#endif