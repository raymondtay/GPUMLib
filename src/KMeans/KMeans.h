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

#ifndef KMeans_h
#define KMeans_h

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

#include <iostream>
#include <ctime>

#include <cula.h>
#include <cublas.h>

#define imax(X, Y)  ((X) > (Y) ? (X) : (Y))
#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))
#define round(x) (x<0?ceil((x)-0.5):floor((x)+0.5))

using namespace std;

typedef std::pair<int, float> my_pair;

namespace GPUMLib {

//! \addtogroup kmeans KMeans Clustering Functions class
//! @{

//! Represents a clustering algorithm using the K-Means technique, implemented in CUDA.
class KMeans{

private:

	void arrayShuffle(int *array, int length);
	int randint(int lowest, int highest);

	CudaStream streamChanged;
	DeviceAccessibleVariable<int> changed_var;
	DeviceAccessibleVariable<bool> changed_bool;

	unsigned int seed;

public:

	//! Base class for the KMeans algorithm.
	KMeans();
	~KMeans();

	//! Executes a clustering algorithm, using the triangle inequality property.
	//! \param Input matrix to apply clustering.
	//! \param kneighbours number of clusters desired.
	//! \return Matrix with the selected cluster centroids.
	DeviceMatrix<float> Execute_TI(DeviceMatrix<float> &Input, int kneighbours);

	//! Executes the standard K-Means algorithm.
	//! \param Input matrix to apply clustering.
	//! \param kneighbours number of clusters desired.
	//! \return Matrix with the selected cluster centroids.
	DeviceMatrix<float> Execute(DeviceMatrix<float> &Input, int kneighbours);

	void SetSeed(unsigned int seed){
		this->seed = seed;
	}

	unsigned int GetSeed(){
		return seed;
	}
};

//! @}

}

#endif