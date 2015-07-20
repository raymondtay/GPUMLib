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

#ifndef Datasetmanager_h
#define Datasetmanager_h

#include "../../common/CudaDefinitions.h"
#include "../../memory/HostMatrix.h"

#include <assert.h>

#include <iostream> 
#include <stdio.h>
#include <cmath>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sstream>

#include <map>
#include <iomanip>
#include <fstream>

using namespace std;
using namespace GPUMLib;

double randDouble(double low, double high);

/*Calculate the root mean square error of outputs against targets*/
double rmse_error(HostMatrix<float> &Target,HostMatrix<float> &Output);

/*Count number of miscalculated outputs against targets*/
int error_calc(HostMatrix<float> &Target,HostMatrix<float> &Output);

/*Read file into matrix X for features and Y for targets, also attributes numbers to classes and 
creates a reverse class lookup to retrieve the names*/
void readFile(string filename, HostMatrix<float> &X, HostMatrix<float> &Y, std::map<string,int> &Classes,std::map<int,string> &ClassesLookup);

/*Read file into matrix X for features and Y for targets*/
void readFile(string filename, HostMatrix<float> &X, HostMatrix<float> &Y);

/*Normalizes input matrix; Ni = (Xi - Mean) / Std*/
void normalize(HostMatrix<float> &X);

/*Retrieves matrix to be used as test using holdout method, 1/3 for test 2/3 for training*/
HostMatrix<float> holdOutTest(HostMatrix<float> &X);

/*Retrieves matrix to be used as train using holdout method, 1/3 for test 2/3 for training*/
HostMatrix<float> holdOutTrain(HostMatrix<float> &X);

/*Retrieves matrix to be used as test using crossvalidation method*/
HostMatrix<float> crossvalidationTest(HostMatrix<float> &X, int folds, int fold_number);

/*Retrieves matrix to be used as train using crossvalidation method*/
HostMatrix<float> crossvalidationTrain(HostMatrix<float> &X, int folds, int fold_number);

/*Display the classifier evaluation measures*/
void measures(int correct_instances,int total_instances, int incorrect_instances, int** confusionMatrix, std::map<string,int> Classes, std::map<int,string> ClassesLookup, ofstream &XMLOutput);
void measures(int correct_instances,int total_instances, int incorrect_instances, int** confusionMatrix, std::map<string,int> Classes, std::map<int,string> ClassesLookup);
#endif