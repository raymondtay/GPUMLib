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

KERNEL CenterAttribution(cudafloat *Output, int output_width, int *attrib_center){

	int bx = blockIdx.x;
    int by = blockIdx.y;

	int idnx = blockIdx.x*blockDim.x + threadIdx.x;
	int idny = blockIdx.y*blockDim.y + threadIdx.y;

	double min_tmp = Output[idny*output_width];
	int idx = 0;

	for(int j = 0; j < output_width; j++){
		if(Output[idny*output_width+j] < min_tmp){
			min_tmp = Output[idny*output_width+j];
			idx = j;
		}
	}

	attrib_center[idny] = idx;
}

extern "C" void KernelCenterAttribution(cudafloat *Output, int output_height, int output_width, int *attrib_center)
{
	int blockSize = 16;

	int wBlocks = 1;
	int hBlocks = output_height/blockSize + ((output_height%blockSize == 0)?0:1);

	dim3 grid(wBlocks,hBlocks);
	dim3 threads(blockSize,blockSize);
	CenterAttribution<<<grid,threads>>>(Output, output_width, attrib_center);
}