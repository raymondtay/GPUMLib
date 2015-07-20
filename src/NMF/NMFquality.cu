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

#include "NMFkernels.h"

namespace GPUMLib {

//! \addtogroup nmfkernels Non-negative Matrix Factorization kernels
//! @{

template <int blockSize> KERNEL NMFquality(cudafloat * V, cudafloat * WH, int n, cudafloat * quality) {
	extern __shared__ cudafloat sum[];

	sum[threadIdx.x] = CUDA_VALUE(0.0);
	for(int k = threadIdx.x; k < n; k += blockSize) {
		cudafloat wh = WH[k];
		sum[threadIdx.x] += (V[k] * log10(wh + SMALL_VALUE_TO_ADD_DENOMINATOR) - wh);
	}
	__syncthreads();

	if (blockSize >= 1024) {
		if (threadIdx.x < 512) sum[threadIdx.x] += sum[threadIdx.x + 512];
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (threadIdx.x < 256) sum[threadIdx.x] += sum[threadIdx.x + 256];
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threadIdx.x < 128) sum[threadIdx.x] += sum[threadIdx.x + 128];
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadIdx.x < 64) sum[threadIdx.x] += sum[threadIdx.x + 64];
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		volatile cudafloat * _sum = sum;

		if (blockSize >= 64) _sum[threadIdx.x] += _sum[threadIdx.x + 32];
		if (blockSize >= 32) _sum[threadIdx.x] += _sum[threadIdx.x + 16];
		if (blockSize >= 16) _sum[threadIdx.x] += _sum[threadIdx.x + 8];
		if (blockSize >= 8) _sum[threadIdx.x] += _sum[threadIdx.x + 4];
		if (blockSize >= 4) _sum[threadIdx.x] += _sum[threadIdx.x + 2];
		if (blockSize >= 2) _sum[threadIdx.x] += _sum[threadIdx.x + 1];
	
		if (threadIdx.x == 0) *quality = sum[0];
	}
}

void KernelNMFquality(int blockSize, cudafloat * V, cudafloat * WH, int n, cudafloat * quality) {
	switch(blockSize) {
		#ifdef FERMI
		case 1024:
			NMFquality<1024><<<1, blockSize, blockSize * sizeof(cudafloat)>>>(V, WH, n, quality);
			break;
		#endif
		case 512:
			NMFquality<512><<<1, blockSize, blockSize * sizeof(cudafloat)>>>(V, WH, n, quality);
			break;
		case 256:
			NMFquality<256><<<1, blockSize, blockSize * sizeof(cudafloat)>>>(V, WH, n, quality);
			break;
		case 128:
			NMFquality<128><<<1, blockSize, blockSize * sizeof(cudafloat)>>>(V, WH, n, quality);
			break;
		case 64:
			NMFquality<64><<<1, blockSize, blockSize * sizeof(cudafloat)>>>(V, WH, n, quality);
			break;
		case 32:
			NMFquality<32><<<1, blockSize, blockSize * sizeof(cudafloat)>>>(V, WH, n, quality);
			break;
		case 16:
			NMFquality<16><<<1, blockSize, blockSize * sizeof(cudafloat)>>>(V, WH, n, quality);
			break;
		case 8:
			NMFquality<8><<<1, blockSize, blockSize * sizeof(cudafloat)>>>(V, WH, n, quality);
			break;
		case 4:
			NMFquality<4><<<1, blockSize, blockSize * sizeof(cudafloat)>>>(V, WH, n, quality);
			break;
		case 2:
			NMFquality<2><<<1, blockSize, blockSize * sizeof(cudafloat)>>>(V, WH, n, quality);
			break;
		case 1:
			NMFquality<1><<<1, blockSize, blockSize * sizeof(cudafloat)>>>(V, WH, n, quality);
			break;
	}
}

//! @}

}