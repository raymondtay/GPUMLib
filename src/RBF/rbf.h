/*
	Ricardo Quintas is a MSc Student at the University of Coimbra, Portugal
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

//! \addtogroup rbffunctions Radial Basis Functions Network functions.
//! @{

//! Functions used to create a radial basis functions network that can be trained using the CUDA implementation of the Radial Basis Functions algorithm.
#ifndef RBF_h
#define RBF_h

#include <vector>
#include <algorithm>

#include "../Common/CudaDefinitions.h"
#include "../memory/HostMatrix.h"
#include "../memory/DeviceMatrix.h"

#include "utils.h"
#include "RBFkernels.h"

#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))

using namespace std;

unsigned int seed = (unsigned)time(0);


float *c_width;

float beta = 3;


typedef std::pair<int, float> my_pair;

//! Shuffles an array.
//! \param[in] array Array to be shuffled.
//! \param[in] length Size of the array.
void arrayShuffle(int *array, int length) {
	for(int i = 0; i < length; i++) {
		int r = rand() % length;
		int t = array[r];
		array[r] = array[i];
		array[i] = t;
	}
}

//! Sorting predicate, for use with the sort function.
bool sort_pred(const my_pair& left, const my_pair& right)
{
	return left.second < right.second;
}


//! Generates a random Integer between lowest and highest values.
int randint(int lowest, int highest){

	srand(seed);

	int range=(highest-lowest)+1;

	return lowest+int(range*rand()/(RAND_MAX + 1.0));
}

//! Calculate the euclidian distance between a row and a column on two given matrixes.
//! \param[in] A Matrix A.
//! \param[in] idxA Row in matrix A.
//! \param[in] B Matrix B.
//! \param[in] idxB Column in matrix B.
//! \return Euclidian distance.
double euclidian_distance(const Eigen::MatrixXd &A, int idxA, const Eigen::MatrixXd &B, int idxB){

	int i;

	double sum = 0;

	double a;
	double b;

	for(i = 0; i < B.rows(); i++){

		a = A(idxA,i);
		b = B(i,idxB);

		sum = sum + pow( a - b , 2);

	}

	return sqrt(sum);
}

//! Calculates the activation of a basis function for a given row and column in matrix A and B.
//! \param[in] A Matrix A.
//! \param[in] idxA Row in matrix A.
//! \param[in] B Matrix B.
//! \param[in] idxB Column in matrix B.
//! \param[in] f Function to use to calculate distance.
//! \return Value of activation function.
float basis_function(const Eigen::MatrixXd &A, int idxA, const Eigen::MatrixXd &B, int idxB, 
					 double (*f)(const Eigen::MatrixXd&, int,const Eigen::MatrixXd&, int)){

						 //return exp(-c_width[idxA] * pow(f(A,idxA,B,idxB),2));

						 return (float) exp(-(pow(f(A,idxA,B,idxB),2)/pow(c_width[idxA],2)));

}

//! Generates an activation matrix between all Input samples and centers of rbf network.
//! \param[in] Input Matrix with data samples.
//! \param[in] Centers Centers of rbf network.
//! \return Matrix with all activation values between samples and centers of rbf network.
Eigen::MatrixXd activation(const Eigen::MatrixXd &Input, const Eigen::MatrixXd &Centers){

	int i, j;

	Eigen::MatrixXd Output(Input.cols(),Centers.rows()); 

	for(i = 0; i < Centers.rows(); i++){

		for(j = 0; j < Input.cols() ; j++){

			Output(j,i) = basis_function(Centers,i,Input,j,euclidian_distance);

		}

	}

	return Output;
}

//! Trains an RBF network.
//! \param[in] Input Matrix with data samples.
//! \param[in] Target Targets for each sample.
//! \param[in] Centers Centers of the RBF network.
//! \return Weights of trained network.
Eigen::MatrixXd trainRBF(const Eigen::MatrixXd &Input,const Eigen::MatrixXd &Target,const Eigen::MatrixXd &Centers){

	Eigen::MatrixXd G = activation(Input,Centers);
	Eigen::MatrixXd Aplus = UTILS::pseudoinverse(G);

	return Aplus*Target.transpose();
}

//! Tests an RBF network.
//! \param[in] Input Matrix with data samples.
//! \param[in] Centers Centers of rbf network.
//! \param[in] Weights Weights of the trained RBF network.
//! \return Results of the application of the RBF network.
Eigen::MatrixXd testRBF(const Eigen::MatrixXd &Input,const Eigen::MatrixXd &Centers, const Eigen::MatrixXd &Weights){
	Eigen::MatrixXd G = activation(Input,Centers);

	return  G*Weights;
}

//! Add a center to the Center matrix.
//! \param[in] X Matrix with data samples.
//! \param[in] Centers Centers of rbf network.
//! \param[in] index Index of the sample to add to the Centers.
//! \return Center matrix with additional entry.
Eigen::MatrixXd addCenter(const Eigen::MatrixXd &X,const Eigen::MatrixXd &Centers, int index){

	Eigen::MatrixXd aux = UTILS::resize(Centers,Centers.rows()+1,Centers.cols());

	for(int i = 0; i < X.rows(); i++)
		aux(aux.rows()-1,i) = X(i,index);

	return aux;
}

//! Function that generates and calculates the RBF network, for the given Input samples.
//! \param[in] Input Matrix with data samples.
//! \param[in] Target Targets for each sample.
//! \param[in] network_size Number of centers to be used in the RBF network.
//! \param[in] number_neighbours Number of neighbours used to estimate the width of the gaussian function for each center.
//! \param[out] OutWeights Calculated weights for the RBF network.
//! \param[out] OutCenters Calculated centers for the RBF network.
void rbf(Eigen::MatrixXd Input,Eigen::MatrixXd Target,int network_size, int number_neighbours,Eigen::MatrixXd &OutWeights,Eigen::MatrixXd &OutCenters){

	int i,j;

	Eigen::MatrixXd Output(Input.rows(),1);

	c_width = (float*) malloc(sizeof(float)*network_size);
	memset(c_width,beta,sizeof(float)*network_size);

	/*Random initialization of centers*/

	int *used;
	used = (int*) malloc(sizeof(int)*Target.cols());

	for(i = 0; i < Target.cols(); i++) { 
		used[i]= i;
	}
	arrayShuffle(used, Target.cols());

	Eigen::MatrixXd Centers = Eigen::MatrixXd(1,Input.rows());

	for(int i = 0; i < Input.rows(); i++)
		Centers(0,i) = Input(i,used[0]);

	for(i = 1; i < network_size; i++){
		Centers = addCenter(Input,Centers,used[i]);
	}

	free(used);

	/*Selection of centers with k-means*/
	bool changed = true;
	int *count = (int*) malloc(sizeof(int)*Centers.rows());

	HostMatrix<float> host_X = HostMatrix<float>(Input.rows(),Input.cols());

	int k = 0;
	for(i = 0; i < Input.rows(); i++){
		for(j = 0; j < Input.cols(); j++){
			host_X.Pointer()[k] = Input(i,j);
			k++;
		}
	}

	DeviceMatrix<float> device_X = DeviceMatrix<float>(host_X);

	HostMatrix<float> host_Centers = HostMatrix<float>(Centers.rows(),Centers.cols());

	k = 0;
	for(i = 0; i < Centers.rows(); i++){
		for(j = 0; j < Centers.cols(); j++){
			host_Centers.Pointer()[k] = Centers(i,j);
			k++;
		}
	}

	DeviceMatrix<float> device_Centers = DeviceMatrix<float>(host_Centers);

	HostMatrix<float> host_output2 = HostMatrix<float>(Input.cols(),Centers.rows());
	DeviceMatrix<float> device_output2 = DeviceMatrix<float>(host_output2);

	int *attrib_center;
	attrib_center = (int*) malloc(sizeof(int)*host_output2.Rows());

	int *device_attrib_center;
	cudaMalloc((void**)&device_attrib_center, sizeof(int)*host_output2.Rows());

	while(changed){

		host_Centers = HostMatrix<float>(Centers.rows(),Centers.cols());

		k = 0;
		for(i = 0; i < Centers.rows(); i++){
			for(j = 0; j < Centers.cols(); j++){
				host_Centers.Pointer()[k] = Centers(i,j);
				k++;
			}
		}

		device_Centers = DeviceMatrix<float>(host_Centers);

		/*Check distances between data and Centers*/
		KernelEuclidianDistance(device_output2.Pointer(), device_output2.Rows(), device_output2.Columns(), device_X.Pointer(), device_X.Columns(), device_Centers.Pointer(), device_Centers.Columns());

		/*Attribution of samples to centers*/
		memset(attrib_center,0,sizeof(int)*host_output2.Rows());
		cudaMemset(device_attrib_center,0,sizeof(int)*host_output2.Rows());

		KernelCenterAttribution(device_output2.Pointer(),device_output2.Rows(), device_output2.Columns(),device_attrib_center);
		cudaMemcpy(attrib_center, device_attrib_center, sizeof(int)*host_output2.Rows(), cudaMemcpyDeviceToHost);


		changed = false;
		memset(count,0,sizeof(int)*host_output2.Columns());

		Eigen::MatrixXd newCenters = Eigen::MatrixXd(network_size,Input.rows());
		newCenters.setZero();

		/*Copy data to new centers, averaging data*/
		for(i = 0; i < host_output2.Rows(); i++){

			for(j = 0; j < newCenters.cols(); j++){

				newCenters(attrib_center[i],j) = (newCenters(attrib_center[i],j) * count[attrib_center[i]] + Input(j,i))/(count[attrib_center[i]]+1);
			}

			count[attrib_center[i]] += 1;

		}

		/*Check if centers changed*/


		for(i = 0; i < Centers.rows(); i++){
			if(euclidian_distance(Centers,i,newCenters.transpose(),i) > 0){
				changed = true;
				break;
			}
		}

		Centers = newCenters;

	}

	free(attrib_center);

	free(count);

	/*Adjust width using mean of distance to neighbours*/
	for(i = 0; i < Centers.rows(); i++){

		std::vector<std::pair<int,float>> tmp;

		for(j = 0; j < Centers.rows(); j++){
			tmp.push_back(std::pair<int,float>(j,euclidian_distance(Centers,i,Centers.transpose(),j)));
		}

		std::sort(tmp.begin(), tmp.end(), sort_pred);

		c_width[i] = 0;

		for(j=0;j<=number_neighbours;j++){
			std::pair<int,float> aux = tmp[j];
			c_width[i] += aux.second;
		}

		c_width[i] = c_width[i]/number_neighbours;
	}

	/*Training*/
	Eigen::MatrixXd Weights = trainRBF(Input,Target,Centers);


	/*Return Weights and Centers*/
	OutWeights = Weights;		
	OutCenters = Centers;

}

#endif

//! @}