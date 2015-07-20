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

#include "RadialBasisFunction.h"

namespace GPUMLib {

RadialBasisFunction::RadialBasisFunction(int network_size, int number_neighbours, float scaling_factor, int NumClasses){

	this->network_size = network_size;
	this->number_neighbours = number_neighbours;
	this->seed = (unsigned)time(0);
	this->scaling_factor = scaling_factor;

	this->NumClasses = NumClasses;

	this->start = 0;
	for(int i = 0; i < 4; i++) times[i] = 0;

	device_c_width = DeviceArray<float>(network_size);

}

RadialBasisFunction::~RadialBasisFunction(){

}

HostMatrix<float> RadialBasisFunction::Test(HostMatrix<float> &Input){

	DeviceMatrix<float> dev_input(Input);

	DeviceMatrix<float> device_activ_matrix(dev_input.Rows(),dCenters.Rows(),ColumnMajor);

	KernelActivationMatrix(device_activ_matrix.Pointer(),dev_input.Pointer(),dCenters.Pointer(),dev_input.Columns(),dCenters.Columns(),device_activ_matrix.Columns(),device_activ_matrix.Rows(),scaling_factor,device_c_width.Pointer());

	DeviceMatrix<float> d_Output(device_activ_matrix.Rows(),dWeights.Columns(),ColumnMajor);
	device_activ_matrix.Multiply(device_activ_matrix,dWeights,d_Output);
	HostMatrix<float> Output(d_Output);

	return Output;	
}

DeviceMatrix<float> RadialBasisFunction::Test(DeviceMatrix<float> &Input){

	DeviceMatrix<float> device_activ_matrix(Input.Rows(),dCenters.Rows(),ColumnMajor);

	KernelActivationMatrix(device_activ_matrix.Pointer(),Input.Pointer(),dCenters.Pointer(),Input.Columns(),dCenters.Columns(),device_activ_matrix.Columns(),device_activ_matrix.Rows(),scaling_factor,device_c_width.Pointer());

	DeviceMatrix<float> d_Output(device_activ_matrix.Rows(),dWeights.Columns(),ColumnMajor);
	device_activ_matrix.Multiply(device_activ_matrix,dWeights,d_Output);

	return d_Output;
}

DeviceMatrix<float> RadialBasisFunction::Test(DeviceMatrix<float> &Input,DeviceMatrix<float> &Centers,DeviceMatrix<float> &Weights, DeviceArray<float> &Widths){

	DeviceMatrix<float> device_activ_matrix(Input.Columns(),Centers.Rows(),ColumnMajor);

	KernelActivationMatrix(device_activ_matrix.Pointer(),Input.Pointer(),Centers.Pointer(),Input.Columns(),Centers.Columns(),device_activ_matrix.Columns(),device_activ_matrix.Rows(),scaling_factor,Widths.Pointer());

	DeviceMatrix<float> d_Output(device_activ_matrix.Rows(),Weights.Columns(),ColumnMajor);
	device_activ_matrix.Multiply(device_activ_matrix,Weights,d_Output);

	return d_Output;

}

void RadialBasisFunction::Train(HostMatrix<float> &Input, HostMatrix<float> &Target){

	//std::cout << "Training" << std::endl;

	//	c_width = (float*) malloc(sizeof(float)*network_size);
	//	memset(c_width,0,sizeof(float)*network_size);

	DeviceMatrix<float> device_X(Input);

	//std::cout << "KMeans" << std::endl;	
	clock_t initialTime = clock();
	KMeans KM;
	KM.SetSeed(seed);
	dCenters = KM.Execute(device_X,network_size);

	cudaThreadSynchronize();
	times[0] = (clock() - initialTime);

	//std::cout << "Adjust Widths" << std::endl;
	/*Adjust width using mean of distance to neighbours*/
	initialTime = clock();
	AdjustWidths(number_neighbours);

	cudaThreadSynchronize();
	times[1] = (clock() - initialTime);

	/*Training weights and scaling factor*/
	HostMatrix<float> TargetArr(Target.Rows(),NumClasses);
	memset(TargetArr.Pointer(),0,sizeof(float)*TargetArr.Elements());

	for(int i = 0; i < Target.Rows(); i++){
		TargetArr(i,((int)Target(i,0)-1)) = 1;
	}

	DeviceMatrix<float> d_Target(TargetArr);

	//std::cout << "Calculating Weights" << std::endl;

	initialTime = clock();

	DeviceMatrix<float> device_activ_matrix(device_X.Rows(),dCenters.Rows(),ColumnMajor);

	KernelActivationMatrix(device_activ_matrix.Pointer(),device_X.Pointer(),dCenters.Pointer(),device_X.Columns(),dCenters.Columns(),device_activ_matrix.Columns(),device_activ_matrix.Rows(),scaling_factor,device_c_width.Pointer());

	DeviceMatrix<float> d_Aplus = UTILS::pseudoinverse(device_activ_matrix);

	dWeights = DeviceMatrix<float>(d_Aplus.Rows(),d_Target.Columns());

	d_Aplus.Multiply(d_Aplus,d_Target,dWeights);


	/*Return Weights and Centers*/
	cudaThreadSynchronize();
	times[2] = (clock() - initialTime);

	// cudaMemcpy(c_width,device_c_width.Pointer(),sizeof(float)*device_c_width.Length(),cudaMemcpyDeviceToHost);
	//	this->Weights = HostMatrix<float>(dWeights);		
	//	this->Centers = HostMatrix<float>(dCenters);

}

void RadialBasisFunction::AdjustWidths(int rneighbours){
	DeviceMatrix<float> device_output2(dCenters.Rows(),dCenters.Rows());
	KernelCalculateDistance(device_output2.Pointer(), dCenters.Pointer(), dCenters.Pointer(),dCenters.Columns(),dCenters.Columns(),device_output2.Columns(),device_output2.Rows());	
	cudaMemset(device_c_width.Pointer(),0,sizeof(float)*device_c_width.Length());
	KernelAdjustWidths(device_output2.Pointer(),device_output2.Rows(), device_output2.Columns(),rneighbours,device_c_width.Pointer());
}

}
