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

#include "KMeans.h"
#include "kmeanskernels.h"

namespace GPUMLib {

KMeans::KMeans(){
	this->seed = (unsigned) time(0);

	changed_var.Value() = 1;
	changed_bool.Value() = false;
}

KMeans::~KMeans(){
}

void KMeans::arrayShuffle(int *array, int length) {

	srand(this->seed);

	for(int i = 0; i < length; i++) {
		int r = rand() % length;
		int t = array[r];
		array[r] = array[i];
		array[i] = t;
	}
}

bool sort_pred(const my_pair& left, const my_pair& right)
{
	return left.second < right.second;
}


int KMeans::randint(int lowest, int highest){

	srand(this->seed);

	int range=(highest-lowest)+1;

	return lowest+int(range*rand()/(RAND_MAX + 1.0));
}


DeviceMatrix<float> KMeans::Execute_TI(DeviceMatrix<float> &Input, int kneighbours){

	int i; //,j;

	/*Random initialization of centers*/

	HostArray<int> used(Input.Rows());

	for(i = 0; i < Input.Rows(); i++) { 
		used[i]= i;
	}

	arrayShuffle(used.Pointer(), used.Length());

	DeviceMatrix<float> dCenters(kneighbours,Input.Columns());

	for(int i = 0; i < kneighbours; i++){
		cudaMemcpy(&(dCenters.Pointer()[i*dCenters.Columns()]),&(Input.Pointer()[used[i]*Input.Columns()]),sizeof(float)*Input.Columns(),cudaMemcpyDeviceToDevice);
	}

	/*Selection of centers with k-means*/
	bool changed = true;

	DeviceArray<float> UpperBounds(Input.Rows());
	DeviceArray<int> CenterAttrib(Input.Rows());

	DeviceMatrix<float> DistanceBeetweenCenters(dCenters.Rows(),dCenters.Rows());

	KernelEuclidianDistance(DistanceBeetweenCenters.Pointer(),dCenters.Pointer(),dCenters.Pointer(),dCenters.Columns(),dCenters.Columns(),DistanceBeetweenCenters.Columns(),DistanceBeetweenCenters.Rows());


	DeviceMatrix<float> InitialDistances(Input.Rows(),dCenters.Rows());
	DeviceMatrix<float> Buffer(Input.Rows(),dCenters.Rows());
	DeviceMatrix<float> LowerBounds(Input.Rows(),dCenters.Rows());

	KernelEuclidianDistance(InitialDistances.Pointer(),Input.Pointer(),dCenters.Pointer(),Input.Columns(),dCenters.Columns(),InitialDistances.Columns(),InitialDistances.Rows());
	cudaMemcpy(LowerBounds.Pointer(),InitialDistances.Pointer(),sizeof(float)*InitialDistances.Elements(),cudaMemcpyDeviceToDevice);

	KernelCenterAttribution_Bounds(InitialDistances.Pointer(),InitialDistances.Rows(),InitialDistances.Columns(),CenterAttrib.Pointer(),UpperBounds.Pointer());


	DeviceMatrix<float> NewCenters(dCenters.Rows(),dCenters.Columns());

	//Copy data to new centers, averaging data

	KernelCopyCenters2(NewCenters.Pointer(),NewCenters.Rows(),NewCenters.Columns(),Input.Pointer(),Input.Rows(),CenterAttrib.Pointer());  

	DeviceArray<float> S(dCenters.Rows());

	DeviceArray<bool> R(Input.Rows());
	cudaMemset(R.Pointer(),true,sizeof(bool)*R.Length());


	KernelEuclidianDistance(DistanceBeetweenCenters.Pointer(),dCenters.Pointer(),NewCenters.Pointer(),dCenters.Columns(),NewCenters.Columns(),DistanceBeetweenCenters.Columns(),DistanceBeetweenCenters.Rows());


	bool flag = true;

	while(changed){

		KernelS(DistanceBeetweenCenters.Pointer(),DistanceBeetweenCenters.Rows(),DistanceBeetweenCenters.Columns(),S.Pointer());


		KernelStep3(Input.Pointer(),Input.Rows(),UpperBounds.Pointer(),S.Pointer(),R.Pointer(),CenterAttrib.Pointer(),LowerBounds.Pointer(),DistanceBeetweenCenters.Pointer(),InitialDistances.Pointer(),NewCenters.Pointer(),NewCenters.Rows(),NewCenters.Columns());

		KernelCopyCenters2(NewCenters.Pointer(),NewCenters.Rows(),NewCenters.Columns(),Input.Pointer(),Input.Rows(),CenterAttrib.Pointer());


		changed = false;

		KernelReduce_bool(R.Pointer(),R.Pointer(),R.Length());

		if (cudaStreamQuery(streamChanged) == cudaSuccess) changed_bool.UpdateValueAsync(R.Pointer(), streamChanged);

		if(!changed_bool.Value()){
			changed = true;
		}

		KernelEuclidianDistance(DistanceBeetweenCenters.Pointer(),dCenters.Pointer(),NewCenters.Pointer(),dCenters.Columns(),NewCenters.Columns(),DistanceBeetweenCenters.Columns(),DistanceBeetweenCenters.Rows());
		KernelStep5(Input.Rows(),UpperBounds.Pointer(),R.Pointer(),CenterAttrib.Pointer(),LowerBounds.Pointer(),DistanceBeetweenCenters.Pointer(),InitialDistances.Pointer(),NewCenters.Pointer(),NewCenters.Rows(),NewCenters.Columns());

		//replace each center
		dCenters = NewCenters;

	}

	return dCenters;

}


DeviceMatrix<float> KMeans::Execute(DeviceMatrix<float> &device_X, int kneighbours){

	int i; //,j;
	/*Random initialization of centers*/

	int *used;
	used = (int*) malloc(sizeof(int)*device_X.Rows());

	for(i = 0; i < device_X.Rows(); i++) { 
		used[i]= i;
	}
	arrayShuffle(used, device_X.Rows());

	DeviceMatrix<float> device_Centers(kneighbours,device_X.Columns());

	for(int i = 0; i < kneighbours; i++){
		cudaMemcpy(&(device_Centers.Pointer()[i*device_Centers.Columns()]),&(device_X.Pointer()[used[i]*device_X.Columns()]),sizeof(float)*device_X.Columns(),cudaMemcpyDeviceToDevice);
	}

	free(used);

	/*Selection of centers with k-means*/
	bool changed = true;

	DeviceMatrix<float> device_output2(device_X.Rows(),device_Centers.Rows());

	DeviceArray<int>device_attrib_center(device_output2.Rows());
	DeviceArray<int>device_attrib_center_old(device_output2.Rows());
	DeviceArray<int>device_attrib_center_out(device_output2.Rows());

	cudaMemset(device_attrib_center_old.Pointer(),0,sizeof(int)*device_attrib_center_old.Length());

	DeviceMatrix<float> device_newCenters(device_Centers.Rows(),device_Centers.Columns());

	while(changed){
		KernelEuclidianDistance(device_output2.Pointer(),device_X.Pointer(),device_Centers.Pointer(),device_X.Columns(),device_Centers.Columns(),device_output2.Columns(),device_output2.Rows());

		/*Attribution of samples to centers*/
		cudaMemset(device_attrib_center.Pointer(),0,sizeof(int)*device_attrib_center.Length());
		KernelCenterAttribution(device_output2.Pointer(),device_output2.Rows(), device_output2.Columns(),device_attrib_center.Pointer());

		cudaMemset(device_output2.Pointer(),0,sizeof(float)*device_output2.Elements());
		KernelPrepareCenterCopy(device_output2.Pointer(),device_output2.Rows(),device_output2.Columns(),device_attrib_center.Pointer());

		/*Copy data to new centers, averaging data*/
		cudaMemset(device_newCenters.Pointer(),0,sizeof(float)*device_newCenters.Elements());
		KernelCopyCenters(device_newCenters.Pointer(),device_newCenters.Rows(),device_newCenters.Columns(),device_X.Pointer(),device_X.Rows(),device_attrib_center.Pointer(),device_output2.Pointer(),device_output2.Rows(), device_output2.Columns());

		/*Check if centers changed*/
		changed = false;

		KernelReduce2(device_attrib_center_out.Pointer(),device_attrib_center.Pointer(),device_attrib_center_old.Pointer(),device_attrib_center.Length());

		if (cudaStreamQuery(streamChanged) == cudaSuccess) changed_var.UpdateValueAsync(device_attrib_center_out.Pointer(), streamChanged);

		if(changed_var.Value() > 0){
			changed = true;
		}

		cudaMemcpy(device_attrib_center_old.Pointer(),device_attrib_center.Pointer(),sizeof(int)*device_attrib_center.Length(),cudaMemcpyDeviceToDevice);
		device_Centers = device_newCenters; 

	}

	return device_Centers;
}

}