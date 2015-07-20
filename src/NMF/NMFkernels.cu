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

// NMF_AditiveEuclidianDistance kernels

KERNEL UpdateMatrix_AE(cudafloat * X, cudafloat * deltaX1, cudafloat * deltaX2, int elements) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < elements) {
		cudafloat v = X[idx] + (X[idx] / deltaX2[idx]) * (deltaX1[idx] - deltaX2[idx]);
		if (v < CUDA_VALUE(0.0)) v = CUDA_VALUE(0.0);
		X[idx] = v;
	}
}

// NMF_MultiplicativeEuclidianDistance kernels

KERNEL UpdateMatrix_ME(cudafloat * nm, cudafloat * dm, cudafloat * m, int elements) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < elements) m[idx] *= nm[idx] / (dm[idx] + SMALL_VALUE_TO_ADD_DENOMINATOR);
}

// NMF_MultiplicativeDivergence kernels

#ifdef ROW_MAJOR_H
	#define HMATRIX(_ROW, _COL, _R, _M) (H[(_ROW) * (_M) + (_COL)])
#else
	#define HMATRIX(_ROW, _COL, _R, _M) (H[(_COL) * (_R) + (_ROW)])
#endif

#ifdef ROW_MAJOR_W
	#define WMATRIX(_ROW, _COL, _N, _R) (W[(_ROW) * (_R) + (_COL)])
#else
	#define WMATRIX(_ROW, _COL, _N, _R) (W[(_COL) * (_N) + (_ROW)])
#endif

template <int blockSize> KERNEL SumW(cudafloat * W, int n, cudafloat * sumW) {
	extern __shared__ cudafloat w[];

	w[threadIdx.x] = CUDA_VALUE(0.0);
	for(int k = threadIdx.x; k < n; k += blockSize) {
		w[threadIdx.x] += WMATRIX(k, blockIdx.x, n, gridDim.x);
	}
	__syncthreads();

	if (blockSize >= 1024) {
		if (threadIdx.x < 512) w[threadIdx.x] += w[threadIdx.x + 512];
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (threadIdx.x < 256) w[threadIdx.x] += w[threadIdx.x + 256];
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threadIdx.x < 128) w[threadIdx.x] += w[threadIdx.x + 128];
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadIdx.x < 64) w[threadIdx.x] += w[threadIdx.x + 64];
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		volatile cudafloat * _w = w;

		if (blockSize >= 64) _w[threadIdx.x] += _w[threadIdx.x + 32];
		if (blockSize >= 32) _w[threadIdx.x] += _w[threadIdx.x + 16];
		if (blockSize >= 16) _w[threadIdx.x] += _w[threadIdx.x + 8];
		if (blockSize >= 8) _w[threadIdx.x] += _w[threadIdx.x + 4];
		if (blockSize >= 4) _w[threadIdx.x] += _w[threadIdx.x + 2];
		if (blockSize >= 2) _w[threadIdx.x] += _w[threadIdx.x + 1];
	
		if (threadIdx.x == 0) {
			cudafloat sum = w[0];
			if (sum < SMALL_VALUE_TO_ADD_DENOMINATOR) sum = SMALL_VALUE_TO_ADD_DENOMINATOR;

			sumW[blockIdx.x] = sum;
		}
	}
}

void KernelSumW(int blockSize, cudafloat * W, int n, int r, cudafloat * sumW) {
	switch(blockSize) {
		#ifdef FERMI
		case 1024:
			SumW<1024><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(W, n, sumW);
			break;
		#endif
		case 512:
			SumW<512><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(W, n, sumW);
			break;
		case 256:
			SumW<256><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(W, n, sumW);
			break;
		case 128:
			SumW<128><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(W, n, sumW);
			break;
		case 64:
			SumW<64><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(W, n, sumW);
			break;
		case 32:
			SumW<32><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(W, n, sumW);
			break;
		case 16:
			SumW<16><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(W, n, sumW);
			break;
		case 8:
			SumW<8><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(W, n, sumW);
			break;
		case 4:
			SumW<4><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(W, n, sumW);
			break;
		case 2:
			SumW<2><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(W, n, sumW);
			break;
		case 1:
			SumW<1><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(W, n, sumW);
			break;
	}
}

template <int blockSize> KERNEL SumH(cudafloat * H, int m, cudafloat * sumH) {
	extern __shared__ cudafloat h[];

	h[threadIdx.x] = CUDA_VALUE(0.0);
	for(int k = threadIdx.x; k < m; k += blockSize) {
		h[threadIdx.x] += HMATRIX(blockIdx.x, k, gridDim.x, m);
	}
	__syncthreads();

	if (blockSize >= 1024) {
		if (threadIdx.x < 512) h[threadIdx.x] += h[threadIdx.x + 512];
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (threadIdx.x < 256) h[threadIdx.x] += h[threadIdx.x + 256];
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threadIdx.x < 128) h[threadIdx.x] += h[threadIdx.x + 128];
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadIdx.x < 64) h[threadIdx.x] += h[threadIdx.x + 64];
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		volatile cudafloat * _h = h;

		if (blockSize >= 64) _h[threadIdx.x] += _h[threadIdx.x + 32];
		if (blockSize >= 32) _h[threadIdx.x] += _h[threadIdx.x + 16];
		if (blockSize >= 16) _h[threadIdx.x] += _h[threadIdx.x + 8];
		if (blockSize >= 8) _h[threadIdx.x] += _h[threadIdx.x + 4];
		if (blockSize >= 4) _h[threadIdx.x] += _h[threadIdx.x + 2];
		if (blockSize >= 2) _h[threadIdx.x] += _h[threadIdx.x + 1];

		if (threadIdx.x == 0) {
			cudafloat sum = h[0];
			if (sum < SMALL_VALUE_TO_ADD_DENOMINATOR) sum = SMALL_VALUE_TO_ADD_DENOMINATOR;

			sumH[blockIdx.x] = sum;
		}
	}
}

void KernelSumH(int blockSize, cudafloat * H, int r, int m, cudafloat * sumH) {
	switch(blockSize) {
		#ifdef FERMI
		case 1024:
			SumH<1024><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(H, m, sumH);
			break;
		#endif
		case 512:
			SumH<512><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(H, m, sumH);
			break;
		case 256:
			SumH<256><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(H, m, sumH);
			break;
		case 128:
			SumH<128><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(H, m, sumH);
			break;
		case 64:
			SumH<64><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(H, m, sumH);
			break;
		case 32:
			SumH<32><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(H, m, sumH);
			break;
		case 16:
			SumH<16><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(H, m, sumH);
			break;
		case 8:
			SumH<8><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(H, m, sumH);
			break;
		case 4:
			SumH<4><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(H, m, sumH);
			break;
		case 2:
			SumH<2><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(H, m, sumH);
			break;
		case 1:
			SumH<1><<<r, blockSize, blockSize * sizeof(cudafloat)>>>(H, m, sumH);
			break;
	}
}

//#define SW(_R, _C) sw[(_R)][(_C)]
#define SW(_R, _C) (sw[(_C)][(_R)])

#define SVH(_R, _C) svh[(_R)][(_C)]
//#define SVH(_R, _C) (svh[(_C)][(_R)])

//#define SH(_R, _C) sh[(_R)][(_C)]
#define SH(_R, _C) sh[(_C)][(_R)]

#define SVW(_R, _C) svw[(_R)][(_C)]
//#define SVW(_R, _C) svw[(_C)][(_R)]

KERNEL UpdateW_MD(cudafloat * W, cudafloat * H, cudafloat * V, cudafloat * WH, cudafloat * sumH, int n, int m, int r) {
	__shared__ cudafloat SH(32, 32);
	__shared__ cudafloat SVW(32, 32);

	int x = blockIdx.x * 32 + threadIdx.x;	
	int y = blockIdx.y * 32 + threadIdx.y;

	cudafloat sum1 = CUDA_VALUE(0.0);
	cudafloat sum2 = CUDA_VALUE(0.0);

	for(int k = 0; k < m; k += 32) {
		int tx = threadIdx.x + 16;

		if (x < r && threadIdx.y + k < m) {
			int ky = k + threadIdx.y;
			SH(threadIdx.x, threadIdx.y) = HMATRIX(x, ky, r, m);
			SH(tx, threadIdx.y) = (x + 16 < r) ? HMATRIX(x + 16, ky, r, m) : CUDA_VALUE(0.0);
		} else {
			SH(threadIdx.x, threadIdx.y) = CUDA_VALUE(0.0);
			SH(tx, threadIdx.y) = CUDA_VALUE(0.0);
		}

		if (y < n && k + threadIdx.x < m) {
			int idx = (k + threadIdx.x) * n + y;
			SVW(threadIdx.y, threadIdx.x) = (V[idx] / (WH[idx] + SMALL_VALUE_TO_ADD_DENOMINATOR));

			idx += (n << 4);
			SVW(threadIdx.y, tx) = (k + tx < m) ? (V[idx] / (WH[idx] + SMALL_VALUE_TO_ADD_DENOMINATOR)) : CUDA_VALUE(0.0);
		} else {
			SVW(threadIdx.y, threadIdx.x) = CUDA_VALUE(0.0);
			SVW(threadIdx.y, tx) = CUDA_VALUE(0.0);
		}
		__syncthreads();

		for(int i = 0; i < 32; i++) {
			sum1 += SH(threadIdx.x, i) * SVW(threadIdx.y, i);
			sum2 += SH(tx, i) * SVW(threadIdx.y, i);
		}
		__syncthreads();
	}

	if (y < n && x < r) {
		WMATRIX(y, x, n, r) *= (sum1 / sumH[x]);
		x += 16;
		if (x < r) WMATRIX(y, x, n, r) *= (sum2 / sumH[x]);
	}
}

KERNEL UpdateH_MD(cudafloat * H, cudafloat * W, cudafloat * V, cudafloat * WH, cudafloat * sumW, int n, int m, int r) {
	__shared__ cudafloat SW(32, 32);
	__shared__ cudafloat SVH(32, 32);

	int x = blockIdx.x * 32 + threadIdx.x;	
	int y = blockIdx.y * 32 + threadIdx.y;

	cudafloat sum1 = CUDA_VALUE(0.0);
	cudafloat sum2 = CUDA_VALUE(0.0);

	for(int k = 0; k < n; k += 32) {
		int ty = threadIdx.y + 16;

		if (y < r && k + threadIdx.x < n) {
			int kx = k + threadIdx.x;
			SW(threadIdx.x, threadIdx.y) = WMATRIX(kx, y, n, r);
			SW(threadIdx.x, ty) = (y + 16 < r) ? WMATRIX(kx, y + 16, n, r) : CUDA_VALUE(0.0);
		} else {
			SW(threadIdx.x, threadIdx.y) = CUDA_VALUE(0.0);
			SW(threadIdx.x, ty) = CUDA_VALUE(0.0);
		}
		
		if (x < m  && k + threadIdx.y < n) {
			int idx = x * n + (k + threadIdx.y);
			SVH(threadIdx.y, threadIdx.x) = V[idx] / (WH[idx] + SMALL_VALUE_TO_ADD_DENOMINATOR);

			idx += 16;
			SVH(ty, threadIdx.x) = (k + ty < n) ? (V[idx] / (WH[idx] + SMALL_VALUE_TO_ADD_DENOMINATOR)) : CUDA_VALUE(0.0);
		} else {
			SVH(threadIdx.y, threadIdx.x) = CUDA_VALUE(0.0);
			SVH(ty, threadIdx.x) = CUDA_VALUE(0.0);
		}
		__syncthreads();

		for(int i = 0; i < 32; i++) {
			sum1 += SW(i, threadIdx.y) * SVH(i, threadIdx.x);
			sum2 += SW(i, ty) * SVH(i, threadIdx.x);
		}
		__syncthreads();
	}

	if (y < r && x < m) {
		HMATRIX(y, x, r, m) *= (sum1 / sumW[y]);
		y += 16;
		if (y < r) HMATRIX(y, x, r, m) *= (sum2 / sumW[y]);
	}
}

//! @}

}