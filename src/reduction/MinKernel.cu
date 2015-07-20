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

#include <limits>
#include "reduction.h"

namespace GPUMLib {

template <int blockSize> KERNEL Min(cudafloat * inputs, cudafloat * output, int numInputs) {
	extern __shared__ cudafloat minvalue[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cudafloat value = MAX_CUDAFLOAT;
	if (idx < numInputs) value = inputs[idx];

	minvalue[threadIdx.x] = value;
	__syncthreads();

	if (blockSize >= 1024) {
		if (threadIdx.x < 512 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 512]) minvalue[threadIdx.x] = minvalue[threadIdx.x + 512];
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (threadIdx.x < 256 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 256]) minvalue[threadIdx.x] = minvalue[threadIdx.x + 256];
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threadIdx.x < 128 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 128]) minvalue[threadIdx.x] = minvalue[threadIdx.x + 128];
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadIdx.x < 64 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 64]) minvalue[threadIdx.x] = minvalue[threadIdx.x + 64];
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		volatile cudafloat * _minvalue = minvalue;

		if (blockSize >= 64) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 32]) _minvalue[threadIdx.x] = _minvalue[threadIdx.x + 32];
		}
	
		if (blockSize >= 32) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 16]) _minvalue[threadIdx.x] = _minvalue[threadIdx.x + 16];
		}
	
		if (blockSize >= 16) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 8]) _minvalue[threadIdx.x] = _minvalue[threadIdx.x + 8];
		}
	
		if (blockSize >= 8) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 4]) _minvalue[threadIdx.x] = _minvalue[threadIdx.x + 4];
		}
	
		if (blockSize >= 4) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 2]) _minvalue[threadIdx.x] = _minvalue[threadIdx.x + 2];
		}
	
		if (blockSize >= 2) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 1]) _minvalue[threadIdx.x] = _minvalue[threadIdx.x + 1];
		}
	
		if (threadIdx.x == 0) {
			output[blockIdx.x] = minvalue[0];
		}
	}
}

template <int blockSize> KERNEL MinIndex(cudafloat * inputs, cudafloat * output, int * indexes, int numInputs) {
	extern __shared__ cudafloat minvalue[];

	int * minpos = (int *) (minvalue + blockDim.x);
  
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cudafloat value = MAX_CUDAFLOAT;
	if (idx < numInputs) value = inputs[idx];

	minvalue[threadIdx.x] = value;
	minpos[threadIdx.x] = idx;
	__syncthreads();

	if (blockSize >= 1024) {
		if (threadIdx.x < 512 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 512]) {
			minvalue[threadIdx.x] = minvalue[threadIdx.x + 512];
			minpos[threadIdx.x] = minpos[threadIdx.x + 512];
		}
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (threadIdx.x < 256 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 256]) {
			minvalue[threadIdx.x] = minvalue[threadIdx.x + 256];
			minpos[threadIdx.x] = minpos[threadIdx.x + 256];
		}
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threadIdx.x < 128 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 128]) {
			minvalue[threadIdx.x] = minvalue[threadIdx.x + 128];
			minpos[threadIdx.x] = minpos[threadIdx.x + 128];
		}
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadIdx.x < 64 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 64]) {
			minvalue[threadIdx.x] = minvalue[threadIdx.x + 64];
			minpos[threadIdx.x] = minpos[threadIdx.x + 64];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		volatile cudafloat * _minvalue = minvalue;
		volatile int * _minpos = minpos;

		if (blockSize >= 64) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 32]) {
				_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 32];
				_minpos[threadIdx.x] = _minpos[threadIdx.x + 32];
			}
		}
	
		if (blockSize >= 32) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 16]) {
				_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 16];
				_minpos[threadIdx.x] = _minpos[threadIdx.x + 16];
			}
		}
	
		if (blockSize >= 16) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 8]) {
				_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 8];
				_minpos[threadIdx.x] = _minpos[threadIdx.x + 8];
			}
		}
	
		if (blockSize >= 8) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 4]) {
				_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 4];
				_minpos[threadIdx.x] = _minpos[threadIdx.x + 4];
			}
		}
	
		if (blockSize >= 4) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 2]) {
				_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 2];
				_minpos[threadIdx.x] = _minpos[threadIdx.x + 2];
			}
		}
	
		if (blockSize >= 2) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 1]) {
				_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 1];
				_minpos[threadIdx.x] = _minpos[threadIdx.x + 1];
			}
		}
	
		if (threadIdx.x == 0) {
			output[blockIdx.x] = minvalue[0];
			indexes[blockIdx.x] = minpos[0];
		}
	}
}

template <int blockSize> KERNEL MinSmallArray(cudafloat * inputs, cudafloat * output, int numInputs) {
	extern __shared__ cudafloat minvalue[];

	minvalue[threadIdx.x] = MAX_CUDAFLOAT;
	for(int i = threadIdx.x; i < numInputs; i += blockDim.x) if (minvalue[threadIdx.x] > inputs[i]) minvalue[threadIdx.x] = inputs[i];
	__syncthreads();

	if (blockSize >= 1024) {
		if (threadIdx.x < 512 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 512]) minvalue[threadIdx.x] = minvalue[threadIdx.x + 512];
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (threadIdx.x < 256 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 256]) minvalue[threadIdx.x] = minvalue[threadIdx.x + 256];
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threadIdx.x < 128 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 128]) minvalue[threadIdx.x] = minvalue[threadIdx.x + 128];
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadIdx.x < 64 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 64]) minvalue[threadIdx.x] = minvalue[threadIdx.x + 64];
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		volatile cudafloat * _minvalue = minvalue;

		if (blockSize >= 64) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 32]) _minvalue[threadIdx.x] = _minvalue[threadIdx.x + 32];
		}
	
		if (blockSize >= 32) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 16]) _minvalue[threadIdx.x] = _minvalue[threadIdx.x + 16];
		}
	
		if (blockSize >= 16) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 8]) _minvalue[threadIdx.x] = _minvalue[threadIdx.x + 8];
		}
	
		if (blockSize >= 8) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 4]) _minvalue[threadIdx.x] = _minvalue[threadIdx.x + 4];
		}
	
		if (blockSize >= 4) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 2]) _minvalue[threadIdx.x] = _minvalue[threadIdx.x + 2];
		}
	
		if (blockSize >= 2) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 1]) _minvalue[threadIdx.x] = _minvalue[threadIdx.x + 1];
		}
	
		if (threadIdx.x == 0) {
			output[blockIdx.x] = minvalue[0];
		}
	}
}

template <int blockSize> KERNEL MinSmallArrayIndex(cudafloat * inputs, cudafloat * output, int * minIndex, int numInputs, int * indexes) {
	extern __shared__ cudafloat minvalue[];

	int * minpos = (int *) (minvalue + blockDim.x);
  
	minvalue[threadIdx.x] = MAX_CUDAFLOAT;
	for(int i = threadIdx.x; i < numInputs; i += blockDim.x) {
		if (minvalue[threadIdx.x] > inputs[i]) {
			minvalue[threadIdx.x] = inputs[i];
			if (indexes != nullptr) {
				minpos[threadIdx.x] = indexes[i];
			} else {
				minpos[threadIdx.x] = i;
			}
		}
	}
	__syncthreads();

	if (blockSize >= 1024) {
		if (threadIdx.x < 512 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 512]) {
			minvalue[threadIdx.x] = minvalue[threadIdx.x + 512];
			minpos[threadIdx.x] = minpos[threadIdx.x + 512];
		}
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (threadIdx.x < 256 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 256]) {
			minvalue[threadIdx.x] = minvalue[threadIdx.x + 256];
			minpos[threadIdx.x] = minpos[threadIdx.x + 256];
		}
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threadIdx.x < 128 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 128]) {
			minvalue[threadIdx.x] = minvalue[threadIdx.x + 128];
			minpos[threadIdx.x] = minpos[threadIdx.x + 128];
		}
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadIdx.x < 64 && minvalue[threadIdx.x] > minvalue[threadIdx.x + 64]) {
			minvalue[threadIdx.x] = minvalue[threadIdx.x + 64];
			minpos[threadIdx.x] = minpos[threadIdx.x + 64];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		volatile cudafloat * _minvalue = minvalue;
		volatile int * _minpos = minpos;

		if (blockSize >= 64) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 32]) {
				_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 32];
				_minpos[threadIdx.x] = _minpos[threadIdx.x + 32];
			}
		}
	
		if (blockSize >= 32) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 16]) {
				_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 16];
				_minpos[threadIdx.x] = _minpos[threadIdx.x + 16];
			}
		}
	
		if (blockSize >= 16) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 8]) {
				_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 8];
				_minpos[threadIdx.x] = _minpos[threadIdx.x + 8];
			}
		}
	
		if (blockSize >= 8) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 4]) {
				_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 4];
				_minpos[threadIdx.x] = _minpos[threadIdx.x + 4];
			}
		}
	
		if (blockSize >= 4) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 2]) {
				_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 2];
				_minpos[threadIdx.x] = _minpos[threadIdx.x + 2];
			}
		}
	
		if (blockSize >= 2) {
			if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 1]) {
				_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 1];
				_minpos[threadIdx.x] = _minpos[threadIdx.x + 1];
			}
		}
	
		if (threadIdx.x == 0) {
			output[blockIdx.x] = minvalue[0];
			minIndex[blockIdx.x] = minpos[0];
		}
	}
}

void KernelMin(cudaStream_t stream, int blocks, int blockSize, cudafloat * inputs, cudafloat * output, int numInputs) {
	if (blocks == 1) {
		switch(blockSize) {
			#ifdef FERMI
			case 1024:
				MinSmallArray<1024><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			#endif
			case 512:
				MinSmallArray<512><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 256:
				MinSmallArray<256><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 128:
				MinSmallArray<128><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 64:
				MinSmallArray<64><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 32:
				MinSmallArray<32><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 16:
				MinSmallArray<16><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 8:
				MinSmallArray<8><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 4:
				MinSmallArray<4><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 2:
				MinSmallArray<2><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 1:
				MinSmallArray<1><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
		}
	} else {
		switch(blockSize) {
			#ifdef FERMI
			case 1024:
				Min<1024><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			#endif
			case 512:
				Min<512><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 256:
				Min<256><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 128:
				Min<128><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 64:
				Min<64><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 32:
				Min<32><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 16:
				Min<16><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 8:
				Min<8><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 4:
				Min<4><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 2:
				Min<2><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 1:
				Min<1><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
		}
	}
}

void KernelMinIndexes(cudaStream_t stream, int blocks, int blockSize, cudafloat * inputs, cudafloat * output, int * minIndexes, int numInputs, int * indexes) {
	if (blocks == 1) {
		switch(blockSize) {
			#ifdef FERMI
			case 1024:
				MinSmallArrayIndex<1024><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs, indexes);
				break;
			#endif
			case 512:
				MinSmallArrayIndex<512><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs, indexes);
				break;
			case 256:
				MinSmallArrayIndex<256><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs, indexes);
				break;
			case 128:
				MinSmallArrayIndex<128><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs, indexes);
				break;
			case 64:
				MinSmallArrayIndex<64><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs, indexes);
				break;
			case 32:
				MinSmallArrayIndex<32><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs, indexes);
				break;
			case 16:
				MinSmallArrayIndex<16><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs, indexes);
				break;
			case 8:
				MinSmallArrayIndex<8><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs, indexes);
				break;
			case 4:
				MinSmallArrayIndex<4><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs, indexes);
				break;
			case 2:
				MinSmallArrayIndex<2><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs, indexes);
				break;
			case 1:
				MinSmallArrayIndex<1><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs, indexes);
				break;
		}
	} else {
		switch(blockSize) {
			#ifdef FERMI
			case 1024:
				MinIndex<1024><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs);
				break;
			#endif
			case 512:
				MinIndex<512><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs);
				break;
			case 256:
				MinIndex<256><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs);
				break;
			case 128:
				MinIndex<128><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs);
				break;
			case 64:
				MinIndex<64><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs);
				break;
			case 32:
				MinIndex<32><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs);
				break;
			case 16:
				MinIndex<16><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs);
				break;
			case 8:
				MinIndex<8><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs);
				break;
			case 4:
				MinIndex<4><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs);
				break;
			case 2:
				MinIndex<2><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs);
				break;
			case 1:
				MinIndex<1><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, minIndexes, numInputs);
				break;
		}
	}
}

}