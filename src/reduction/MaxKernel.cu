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

#include "reduction.h"

namespace GPUMLib {

template <int blockSize> KERNEL Max(cudafloat * inputs, cudafloat * output, int numInputs) {
	extern __shared__ cudafloat maxvalue[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cudafloat value = MIN_CUDAFLOAT;
	if (idx < numInputs) value = inputs[idx];

	maxvalue[threadIdx.x] = value;
	__syncthreads();

	if (blockSize >= 1024) {
		if (threadIdx.x < 512 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 512]) maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 512];
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (threadIdx.x < 256 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 256]) maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 256];
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threadIdx.x < 128 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 128]) maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 128];
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadIdx.x < 64 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 64]) maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 64];
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		volatile cudafloat * _maxvalue = maxvalue;

		if (blockSize >= 64) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 32]) _maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 32];
		}
	
		if (blockSize >= 32) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 16]) _maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 16];
		}
	
		if (blockSize >= 16) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 8]) _maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 8];
		}
	
		if (blockSize >= 8) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 4]) _maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 4];
		}
	
		if (blockSize >= 4) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 2]) _maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 2];
		}
	
		if (blockSize >= 2) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 1]) _maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 1];
		}
	
		if (threadIdx.x == 0) {
			output[blockIdx.x] = maxvalue[0];
		}
	}
}

template <int blockSize> KERNEL MaxIndex(cudafloat * inputs, cudafloat * output, int * indexes, int numInputs) {
	extern __shared__ cudafloat maxvalue[];

	int * maxpos = (int *) (maxvalue + blockDim.x);
  
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cudafloat value = MIN_CUDAFLOAT;
	if (idx < numInputs) value = inputs[idx];

	maxvalue[threadIdx.x] = value;
	maxpos[threadIdx.x] = idx;
	__syncthreads();

	if (blockSize >= 1024) {
		if (threadIdx.x < 512 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 512]) {
			maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 512];
			maxpos[threadIdx.x] = maxpos[threadIdx.x + 512];
		}
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (threadIdx.x < 256 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 256]) {
			maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 256];
			maxpos[threadIdx.x] = maxpos[threadIdx.x + 256];
		}
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threadIdx.x < 128 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 128]) {
			maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 128];
			maxpos[threadIdx.x] = maxpos[threadIdx.x + 128];
		}
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadIdx.x < 64 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 64]) {
			maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 64];
			maxpos[threadIdx.x] = maxpos[threadIdx.x + 64];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		volatile cudafloat * _maxvalue = maxvalue;
		volatile int * _maxpos = maxpos;

		if (blockSize >= 64) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 32]) {
				_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 32];
				_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 32];
			}
		}
	
		if (blockSize >= 32) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 16]) {
				_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 16];
				_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 16];
			}
		}
	
		if (blockSize >= 16) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 8]) {
				_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 8];
				_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 8];
			}
		}
	
		if (blockSize >= 8) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 4]) {
				_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 4];
				_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 4];
			}
		}
	
		if (blockSize >= 4) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 2]) {
				_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 2];
				_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 2];
			}
		}
	
		if (blockSize >= 2) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 1]) {
				_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 1];
				_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 1];
			}
		}
	
		if (threadIdx.x == 0) {
			output[blockIdx.x] = maxvalue[0];
			indexes[blockIdx.x] = maxpos[0];
		}
	}
}

template <int blockSize> KERNEL MaxSmallArray(cudafloat * inputs, cudafloat * output, int numInputs) {
	extern __shared__ cudafloat maxvalue[];

	maxvalue[threadIdx.x] = MIN_CUDAFLOAT;
	for(int i = threadIdx.x; i < numInputs; i += blockDim.x) if (maxvalue[threadIdx.x] < inputs[i]) maxvalue[threadIdx.x] = inputs[i];
	__syncthreads();

	if (blockSize >= 1024) {
		if (threadIdx.x < 512 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 512]) maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 512];
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (threadIdx.x < 256 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 256]) maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 256];
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threadIdx.x < 128 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 128]) maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 128];
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadIdx.x < 64 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 64]) maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 64];
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		volatile cudafloat * _maxvalue = maxvalue;

		if (blockSize >= 64) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 32]) _maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 32];
		}
	
		if (blockSize >= 32) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 16]) _maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 16];
		}
	
		if (blockSize >= 16) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 8]) _maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 8];
		}
	
		if (blockSize >= 8) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 4]) _maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 4];
		}
	
		if (blockSize >= 4) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 2]) _maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 2];
		}
	
		if (blockSize >= 2) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 1]) _maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 1];
		}
	
		if (threadIdx.x == 0) {
			output[blockIdx.x] = maxvalue[0];
		}
	}
}

template <int blockSize> KERNEL MaxSmallArrayIndex(cudafloat * inputs, cudafloat * output, int * maxIndex, int numInputs, int * indexes) {
	extern __shared__ cudafloat maxvalue[];

	int * maxpos = (int *) (maxvalue + blockDim.x);
  
	maxvalue[threadIdx.x] = MIN_CUDAFLOAT;
	for(int i = threadIdx.x; i < numInputs; i += blockDim.x) {
		if (maxvalue[threadIdx.x] < inputs[i]) {
			maxvalue[threadIdx.x] = inputs[i];
			if (indexes != nullptr) {
				maxpos[threadIdx.x] = indexes[i];
			} else {
				maxpos[threadIdx.x] = i;
			}
		}
	}
	__syncthreads();

	if (blockSize >= 1024) {
		if (threadIdx.x < 512 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 512]) {
			maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 512];
			maxpos[threadIdx.x] = maxpos[threadIdx.x + 512];
		}
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (threadIdx.x < 256 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 256]) {
			maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 256];
			maxpos[threadIdx.x] = maxpos[threadIdx.x + 256];
		}
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threadIdx.x < 128 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 128]) {
			maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 128];
			maxpos[threadIdx.x] = maxpos[threadIdx.x + 128];
		}
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadIdx.x < 64 && maxvalue[threadIdx.x] < maxvalue[threadIdx.x + 64]) {
			maxvalue[threadIdx.x] = maxvalue[threadIdx.x + 64];
			maxpos[threadIdx.x] = maxpos[threadIdx.x + 64];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		volatile cudafloat * _maxvalue = maxvalue;
		volatile int * _maxpos = maxpos;

		if (blockSize >= 64) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 32]) {
				_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 32];
				_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 32];
			}
		}
	
		if (blockSize >= 32) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 16]) {
				_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 16];
				_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 16];
			}
		}
	
		if (blockSize >= 16) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 8]) {
				_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 8];
				_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 8];
			}
		}
	
		if (blockSize >= 8) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 4]) {
				_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 4];
				_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 4];
			}
		}
	
		if (blockSize >= 4) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 2]) {
				_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 2];
				_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 2];
			}
		}
	
		if (blockSize >= 2) {
			if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 1]) {
				_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 1];
				_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 1];
			}
		}
	
		if (threadIdx.x == 0) {
			output[blockIdx.x] = maxvalue[0];
			maxIndex[blockIdx.x] = maxpos[0];
		}
	}
}

void KernelMax(cudaStream_t stream, int blocks, int blockSize, cudafloat * inputs, cudafloat * output, int numInputs) {
	if (blocks == 1) {
		switch(blockSize) {
			#ifdef FERMI
			case 1024:
				MaxSmallArray<1024><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			#endif
			case 512:
				MaxSmallArray<512><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 256:
				MaxSmallArray<256><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 128:
				MaxSmallArray<128><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 64:
				MaxSmallArray<64><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 32:
				MaxSmallArray<32><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 16:
				MaxSmallArray<16><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 8:
				MaxSmallArray<8><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 4:
				MaxSmallArray<4><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 2:
				MaxSmallArray<2><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 1:
				MaxSmallArray<1><<<1, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
		}
	} else {
		switch(blockSize) {
			#ifdef FERMI
			case 1024:
				Max<1024><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			#endif
			case 512:
				Max<512><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 256:
				Max<256><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 128:
				Max<128><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 64:
				Max<64><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 32:
				Max<32><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 16:
				Max<16><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 8:
				Max<8><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 4:
				Max<4><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 2:
				Max<2><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
			case 1:
				Max<1><<<blocks, blockSize, blockSize * sizeof(cudafloat), stream>>>(inputs, output, numInputs);
				break;
		}
	}
}

void KernelMaxIndexes(cudaStream_t stream, int blocks, int blockSize, cudafloat * inputs, cudafloat * output, int * maxIndexes, int numInputs, int * indexes) {
	if (blocks == 1) {
		switch(blockSize) {
			#ifdef FERMI
			case 1024:
				MaxSmallArrayIndex<1024><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs, indexes);
				break;
			#endif
			case 512:
				MaxSmallArrayIndex<512><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs, indexes);
				break;
			case 256:
				MaxSmallArrayIndex<256><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs, indexes);
				break;
			case 128:
				MaxSmallArrayIndex<128><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs, indexes);
				break;
			case 64:
				MaxSmallArrayIndex<64><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs, indexes);
				break;
			case 32:
				MaxSmallArrayIndex<32><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs, indexes);
				break;
			case 16:
				MaxSmallArrayIndex<16><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs, indexes);
				break;
			case 8:
				MaxSmallArrayIndex<8><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs, indexes);
				break;
			case 4:
				MaxSmallArrayIndex<4><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs, indexes);
				break;
			case 2:
				MaxSmallArrayIndex<2><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs, indexes);
				break;
			case 1:
				MaxSmallArrayIndex<1><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs, indexes);
				break;
		}
	} else {
		switch(blockSize) {
			#ifdef FERMI
			case 1024:
				MaxIndex<1024><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs);
				break;
			#endif
			case 512:
				MaxIndex<512><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs);
				break;
			case 256:
				MaxIndex<256><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs);
				break;
			case 128:
				MaxIndex<128><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs);
				break;
			case 64:
				MaxIndex<64><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs);
				break;
			case 32:
				MaxIndex<32><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs);
				break;
			case 16:
				MaxIndex<16><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs);
				break;
			case 8:
				MaxIndex<8><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs);
				break;
			case 4:
				MaxIndex<4><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs);
				break;
			case 2:
				MaxIndex<2><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs);
				break;
			case 1:
				MaxIndex<1><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int)), stream>>>(inputs, output, maxIndexes, numInputs);
				break;
		}
	}
}

}