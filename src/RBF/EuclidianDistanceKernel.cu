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

#include "../Common/CudaDefinitions.h"
#include "RBFkernels.h"


/* KERNEL Euclidian Distance */
KERNEL EuclidianDistance(cudafloat *Output, int output_height, int output_width, cudafloat *Input, int input_width, cudafloat *Centers, int centers_width){

	int idx = threadIdx.y * output_width + threadIdx.x;     
	
	int bx = blockIdx.x;
    int by = blockIdx.y;

	int idnx = blockIdx.x*blockDim.x + threadIdx.x;
	int idny = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(idnx < output_width && idny < output_height){

			double sum = 0;

			double a;
			double b;
		
			for(int i = 0; i < centers_width; i++){
			
				a = Centers[idnx * centers_width + i];
				b = Input[idny + i * input_width];

				sum = sum + pow( a - b , 2);

			}
				
			Output[idnx + idny * output_width] = sqrt(sum);
	}
}

extern "C" void KernelEuclidianDistance(cudafloat *Output, int output_height, int output_width, cudafloat *Input, int input_width, cudafloat *Centers, int centers_width)
{
	int blockSize = 16;

	int wBlocks = output_width/blockSize + ((output_width%blockSize == 0)?0:1);
	int hBlocks = output_height/blockSize + ((output_height%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	EuclidianDistance<<<grid,threads>>>(Output, output_height, output_width, Input, input_width, Centers, centers_width);
}
