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

//! \addtogroup kmeanskernels KMeans Clustering kernels
//! @{
#ifndef KMeanskernels_h
#define KMeanskernels_h

#include "../common/CudaDefinitions.h"
#include "../memory/DeviceArray.h"
#include "../memory/DeviceMatrix.h"

#define imax(X, Y)  ((X) > (Y) ? (X) : (Y))
#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))
#define round(x) (x<0?ceil((x)-0.5):floor((x)+0.5))

/*Define functions to call CUDA kernels in C program*/

/*Functions for KMEANS standard*/

//! Kernel that calculates the distance between all rows of matrix A in relation to all rows of matrix B, the result is stored in matrix C. Index (i,j) in matrix C is equivalent to the distance between row i in matrix A and row j in matrix B.
//! \param[out] d_C Final matrix with the calculated distances.
//! \param[in] d_A Matrix A.
//! \param[in] d_B Matrix B.
//! \param[in] uiWA Width of matrix A.
//! \param[in] uiWB Width of matrix B.
//! \param[in] uiWC Width of matrix C.
//! \param[in] uiHC Height of matrix C.
extern "C" void KernelEuclidianDistance(cudafloat *d_C, cudafloat* d_A, cudafloat* d_B,int uiWA,int uiWB, int uiWC, int uiHC);

//! Kernel that attributes each sample of the training data to one cluster. Given a distance matrix, it finds the minimum distance and stores the obtained cluster number in the attribution array.
//! \param[in] Output Distance matrix, each row has the distance between a row in the training data and cluster represented by each column.
//! \param[in] output_height Height of the distance matrix (number of training samples).
//! \param[in] output_width Width of the distance matrix (number of clusters).
//! \param[out] attrib_center Output array with an entry for each sample in the training data with the attributed cluster.
extern "C" void KernelCenterAttribution(cudafloat *Output, int output_height, int output_width, int *attrib_center);

//! Kernel that given an array with the attributed clusters for each training sample, stores the indexes of each sample in the output matrix in the corresponding column to its attributed cluster.
//! \param[out] Output Matrix with the indexes of the samples to average for each cluster.
//! \param[in] output_height Height of the index copy matrix (number of training samples).
//! \param[in] output_width Width of the index copy matrix (number of clusters).
//! \param[in] attrib_center Array with an entry for each sample in the training data with the attributed cluster.
extern "C" void KernelPrepareCenterCopy(cudafloat *Output, int output_height, int output_width, int *attrib_center);

//! Kernel that averages all the training samples attributed to a cluster, and stores the result as the new cluster centroid.
//! \param[out] Output Matrix with the new centroids for all clusters.
//! \param[in] output_height Height of the centroid matrix (number of clusters).
//! \param[in] output_width Width of the centroid matrix (number of features).
//! \param[out] Input Matrix with the indexes of the samples to average for each cluster.
//! \param[in] input_height Height of the index copy matrix.
//! \param[in] attrib_center Array with an entry for each sample in the training data with the attributed cluster.
//! \param[in] Indexes indexes.
//! \param[in] idx_height index height.
//! \param[in] idx_width index width.
extern "C" void KernelCopyCenters(cudafloat *Output, int output_height, int output_width, cudafloat *Input,int input_height, int *attrib_center,cudafloat *Indexes, int idx_height, int idx_width);

//! Kernel that compares two arrays and stores the result.
//! \param[out] output Array with the result of the comparation.
//! \param[in] input Array with the attribution of training samples to clusters.
//! \param[in] g_idata_old Array with the old attribution.
//! \param[in] n Length of the array (number of training samples)
extern "C" void KernelReduce2(int *output, int *input, int *g_idata_old,int n);


/*Function for KMEANS with triangle inequality*/

//! Kernel that attributes each sample of the training data to one cluster, and updates the upperbounds. Given a distance matrix, it finds the minimum distance and stores the obtained cluster number in the attribution array.
//! \param[in] Output Distance matrix, each row has the distance between a row in the training data and cluster represented by each column.
//! \param[in] output_height Height of the distance matrix (number of training samples).
//! \param[in] output_width Width of the distance matrix (number of clusters).
//! \param[out] attrib_center Output array with an entry for each sample in the training data with the attributed cluster.
//! \param[out] upperbound Array with the upperbounds for each sample.
extern "C" void KernelCenterAttribution_Bounds(cudafloat *Output, int output_height, int output_width, int *attrib_center, float* upperbound);

//! Kernel that averages all the training samples attributed to a cluster, and stores the result as the new cluster centroid.
//! \param[out] Output Matrix with the new centroids for all clusters.
//! \param[in] output_height Height of the centroid matrix (number of clusters).
//! \param[in] output_width Width of the centroid matrix (number of features).
//! \param[in] Input Matrix with the training samples to average.
//! \param[in] input_width Height of the training matrix.
//! \param[in] attrib_center Array with an entry for each sample in the training data with the attributed cluster.
extern "C" void KernelCopyCenters2(cudafloat *Output, int output_height, int output_width, cudafloat *Input,int input_width, int *attrib_center);

//! Kernel that updates the S value for each cluster.
//! \param[in] Output Matrix with the distances between clusters.
//! \param[in] output_height Height of the distance matrix
//! \param[in] output_width Width of the distance matrix.
//! \param[out] S Array with the S value for each cluster.
extern "C" void KernelS(cudafloat *Output, int output_height, int output_width, float *S);

//! Kernel that calculates the distance between the rows of two matrices, applies the triangle inequality property to minimize the number of calculations.
//! \param[in] Input Matrix with the training data.
//! \param[in] input_height Height of the training matrix.
//! \param[in] Upperbounds Array with the upperbounds for each training sample.
//! \param[in] S Array with the S value for each cluster.
//! \param[in] R Array of booleans with information if training sample changed its cluster attribution.
//! \param[in] CenterAttrib Array with the attribution of samples to clusters.
//! \param[in] LowerBounds Array with the lowerbounds for the training samples.
//! \param[in] DistanceBeetweenCenters Matrix with the distance between centroids.
//! \param[in] InitialDistances Matrix with distances between training samples and cluster centroids.
//! \param[in] NewCenters Matrix with the centroids of the chosen clusters.
//! \param[in] centers_height Height of the centroid matrix.
//! \param[in] centers_width Width of the centroid matrix.
extern "C" void KernelStep3(float*Input,int input_height, float* Upperbounds, float* S,bool* R,int* CenterAttrib,float* LowerBounds,float* DistanceBeetweenCenters,float* InitialDistances, float* NewCenters,int centers_height,int centers_width);

//! Kernel that checks if all values in a given array are true.
//! \param[out] output Array with the result of the comparation.
//! \param[in] input Array of booleans, true for changed data.
//! \param[in] n Length of the array (number of training samples)
extern "C" void KernelReduce_bool(bool *output, bool *input,int n);

//! Kernel that updates the upper and lower bounds. Resets R array. 
//! \param[in] input_height Height of the training matrix.
//! \param[out] Upperbounds Array with the upperbounds for each training sample.
//! \param[out] R Array of booleans with information if training sample changed its cluster attribution.
//! \param[in] CenterAttrib Array with the attribution of samples to clusters.
//! \param[out] LowerBounds Array with the lowerbounds for the training samples.
//! \param[in] DistanceBeetweenCenters Matrix with the distance between centroids.
//! \param[in] InitialDistances Matrix with distances between training samples and cluster centroids.
//! \param[in] NewCenters Matrix with the centroids of the chosen clusters.
//! \param[in] centers_height Height of the centroid matrix.
//! \param[in] centers_width Width of the centroid matrix.
extern "C" void KernelStep5(int input_height, float* Upperbounds, bool* R,int* CenterAttrib,float* LowerBounds,float* DistanceBeetweenCenters,float* InitialDistances, float* NewCenters,int centers_height,int centers_width);


/*Auxiliary functions*/
extern "C" unsigned int nextPow2( unsigned int x );

#endif

//! @}