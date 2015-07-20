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

#include "../reduction/SumWarp.h"
#include "MBPkernels.h"

namespace GPUMLib {

template <int blockSize> KERNEL CalculateRMS(cudafloat * rms, cudafloat * rmsF, int numberPatterns, cudafloat numberPatternsNeurons) {
	extern __shared__ cudafloat shared_rms[];
	
	shared_rms[threadIdx.x] = CUDA_VALUE(0.0);
	for(int p = threadIdx.x; p < numberPatterns; p += blockDim.x) shared_rms[threadIdx.x] += rms[p];
	__syncthreads();

	SumBeforeWarp<blockSize>(shared_rms);

	if (threadIdx.x < 32) {
		SumWarp<blockSize>(shared_rms);

		if (threadIdx.x == 0) {
			cudafloat fRMS = CUDA_SQRT(shared_rms[0] / numberPatternsNeurons) / CUDA_VALUE(2.0);
			if (IsInfOrNaN(fRMS)) fRMS = numberPatternsNeurons;
			*rmsF = fRMS;
		}
	}
}

void KernelCalculateRMS(cudaStream_t stream, int blockSize, cudafloat * rms, cudafloat * rmsOut, int numberPatterns, cudafloat numberPatternsNeurons) {
	switch(blockSize) {
		#ifdef FERMI
		case 1024:
			CalculateRMS<1024><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
			break;
		#endif
		case 512:
			CalculateRMS<512><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
			break;

		case 256:
			CalculateRMS<256><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
			break;

		case 128:
			CalculateRMS<128><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
			break;

		case 64:
			CalculateRMS<64><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
			break;

		case 32:
			CalculateRMS<32><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
			break;

		case 16:
			CalculateRMS<16><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
			break;

		case 8:
			CalculateRMS<8><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
			break;

		case 4:
			CalculateRMS<4><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
			break;

		case 2:
			CalculateRMS<2><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
			break;

		case 1:
			CalculateRMS<1><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
			break;
	}
}

}