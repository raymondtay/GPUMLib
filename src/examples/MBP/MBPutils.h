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

#ifndef MBPutils_h
#define MBPutils_h

#define MNIST_SMALL_DATASET

#include "../common/ConfusionMatrix.h"
#include "../../MBP/BackPropagation.h"
#ifndef GPUMLIB_DBN
#include "../../MBP/MultipleBackPropagation.h"
#endif

using namespace GPUMLib;

void SaveMBPNetwork(const char * filename, BackPropagation * network, const char * trainfilename, const char * testfilename, float rmsStop, int epochs, bool MBP);

#ifdef MNIST_SMALL_DATASET
void AnalyzeTestData(BackPropagation * network, HostMatrix<cudafloat> & inputs, HostMatrix<cudafloat> & desiredOutputs, ConfusionMatrix & cm, ConfusionMatrix & cm_small);
#else
void AnalyzeTestData(BackPropagation * network, HostMatrix<cudafloat> & inputs, HostMatrix<cudafloat> & desiredOutputs, ConfusionMatrix & cm);
#endif

#endif