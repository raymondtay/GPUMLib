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

#ifndef RAN_utils_h
#define RAN_utils_h

#include "../common/CudaDefinitions.h"
#include "../memory/HostMatrix.h"


#include "../memory/DeviceMatrix.h"
#include "rankernels.h"
#include <cula.h>
#include <cublas.h>
#include <cuda.h>

#include <iostream>
#include <string>
#include <fstream>

#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))
/*Utilities for matrix operations*/


void checkStatus(culaStatus status);

using namespace GPUMLib;

namespace UTILS{

	void writeM(std::string desc, HostMatrix<float> Input);
	void writeM(std::string desc, DeviceMatrix<float> Mat);
	void printM(std::string desc, DeviceMatrix<float> Mat, bool Order);
	void printA(std::string desc, DeviceArray<float> Array);
	DeviceMatrix<float> pseudoinverse(DeviceMatrix<float> &Input);
	void pseudoinverse2(DeviceMatrix<float> &Input);


	void printM(std::string desc, HostMatrix<float> Input, bool Order, int Rows);
	void printM(std::string desc, HostMatrix<float> Mat, int Rows);
	void printM(std::string desc, HostMatrix<float> Mat);
	void printM(std::string desc, HostMatrix<float> Mat, bool Order);
	void printM(std::string desc, DeviceMatrix<float> Mat, bool Order);
	void printM(std::string desc, DeviceMatrix<float> Mat, int Rows);


	void printA(std::string desc, HostArray<float> Array);
	void printA(std::string desc, HostArray<float> Array, int Length);

};

#endif
