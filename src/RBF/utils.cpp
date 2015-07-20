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

#include "utils.h"

void checkStatus(culaStatus status)
{
	if(!status)
		return;

	if(status == culaArgumentError)
		printf("Invalid value for parameter %d\n", culaGetErrorInfo());
	else if(status == culaDataError)
		printf("Data error (%d)\n", culaGetErrorInfo());
	else if(status == culaBlasError)
		printf("Blas error (%d)\n", culaGetErrorInfo());
	else if(status == culaRuntimeError)
		printf("Runtime error (%d)\n", culaGetErrorInfo());
	else
		printf("%s\n", culaGetStatusString(status));

	culaShutdown();
	exit(EXIT_FAILURE);
}


namespace UTILS{

	void writeM(std::string desc, HostMatrix<float> Input){

		std::ofstream File(desc.c_str());

		File << desc << "=";
		File << "[";
		for(int i=0;i<Input.Rows(); i++){
			for(int j=0;j<Input.Columns();j++){
				File << Input(i,j) << " ";
			}
			if(i < Input.Rows() - 1)
				File << ";\n";
		}
		File << "];";	

		File.close();
	}

	void writeM(std::string desc, DeviceMatrix<float> Mat){


		HostMatrix<float> Input = HostMatrix<float>(Mat);

		writeM(desc,Input);

	}

	void printM(std::string desc, HostMatrix<float> Input, bool Order, int Rows){

		if(!Order){
			std::cout << desc << " " << Input.Rows() << "x" << Input.Columns() << std::endl;
			std::cout << "[";
			for(int i=0;i<Rows; i++){
				for(int j=0;j<Input.Columns();j++){
					std::cout << Input(i,j) << " ";
				}
				if(i < Rows - 1)
					std::cout << ";" << std::endl;
			}
			std::cout << "]";
			std::cout << std::endl << std::endl;
		}else{
			std::cout << desc << " " << Rows << "x" << Input.Columns() << std::endl;
			std::cout << "[";

			for(int i=0;i<Input.Rows(); i++){
				for(int j=0;j<Input.Columns();j++){
					std::cout << Input.Pointer()[j*Input.Rows()+i] << " ";
				}
				if(i < Input.Rows() - 1)
					std::cout << ";" << std::endl;
			}
			std::cout << "]";
			std::cout << std::endl << std::endl;
		}
	}

	void printM(std::string desc, HostMatrix<float> Mat, int Rows){
		printM(desc,Mat,false,Rows);
	}

	void printM(std::string desc, HostMatrix<float> Mat){
		printM(desc,Mat,false,Mat.Rows());
	}

	void printM(std::string desc, HostMatrix<float> Mat, bool Order){
		printM(desc,Mat,Order,Mat.Rows());
	}

	void printM(std::string desc, DeviceMatrix<float> Mat, bool Order){
		HostMatrix<float> Input = HostMatrix<float>(Mat);
		printM(desc,Input, Order);
	}

	void printM(std::string desc, DeviceMatrix<float> Mat, int Rows){
		HostMatrix<float> Input = HostMatrix<float>(Mat);
		printM(desc,Input,false,Rows);
	}

	void printA(std::string desc, HostArray<float> Array){

		std::cout << desc << " " << Array.Length() << std::endl;
		std::cout << "[";
		for(int i=0;i<Array.Length(); i++){
			std::cout << Array[i] << " ";
			if(i < Array.Length() - 1)
				std::cout << ";";
		}
		std::cout << "]";
		std::cout << std::endl << std::endl;

	}

	void printA(std::string desc, HostArray<int> Array){

		std::cout << desc << " " << Array.Length() << std::endl;
		std::cout << "[";
		for(int i=0;i<Array.Length(); i++){
			std::cout << Array[i] << " ";
			if(i < Array.Length() - 1)
				std::cout << ";";
		}
		std::cout << "]";
		std::cout << std::endl << std::endl;

	}

	void printA(std::string desc, HostArray<float> Array, int Length){

		std::cout << desc << " " << Length << std::endl;
		std::cout << "[";
		for(int i=0;i<Length; i++){
			std::cout << Array[i] << " ";
			if(i < Length - 1)
				std::cout << ";";
		}
		std::cout << "]";
		std::cout << std::endl << std::endl;

	}

	void printA(std::string desc, DeviceArray<float> Array){

		HostArray<float> Input = HostArray<float>(Array);

		printA(desc,Input);

	}

	void printA(std::string desc, DeviceArray<int> Array){

		HostArray<int> Input = HostArray<int>(Array);

		printA(desc,Input);

	}


	DeviceMatrix<float> pseudoinverse(DeviceMatrix<float> &Input){

		int M = Input.Rows();
		int N = Input.Columns();

		/* Setup SVD Parameters */
		int LDA = M;
		int LDU = M;
		int LDVT = N;

		char jobu = 'A';
		char jobvt = 'A';

		DeviceMatrix<float> S(imin(M,N),1,ColumnMajor);
		DeviceMatrix<float> U(LDU,M,ColumnMajor);
		DeviceMatrix<float> VT(LDVT,N,ColumnMajor);

		checkStatus(culaDeviceSgesvd(jobu, jobvt, M, N, Input.Pointer(), LDA, S.Pointer(), U.Pointer(), LDU, VT.Pointer(), LDVT));

		DeviceMatrix<float> d_Sigma = DeviceMatrix<float>(N,M,ColumnMajor);
		cudaMemset(d_Sigma.Pointer(),0,sizeof(float)*d_Sigma.Elements());

		KernelSigmaInverse(d_Sigma.Pointer(),d_Sigma.Rows(),d_Sigma.Columns(),S.Pointer());

		DeviceMatrix<float> sigmainverse = DeviceMatrix<float>(N,M,ColumnMajor);
		DeviceMatrix<float> aux2 = DeviceMatrix<float>(N,M,ColumnMajor);

		cublasSgemm('t', 'n', N, M, N, 1,     VT.Pointer(), LDVT, d_Sigma.Pointer(), N, 0, aux2.Pointer(), N);
		cublasSgemm('n', 't', N, M, M, 1, aux2.Pointer(), N,    U.Pointer(), M, 0, sigmainverse.Pointer(), N);

		return sigmainverse;
	}

}
