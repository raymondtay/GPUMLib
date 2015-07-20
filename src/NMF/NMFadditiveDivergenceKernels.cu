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

#ifdef ROW_MAJOR_H
	#define HMATRIX(_ROW, _COL, _R, _M) (H[(_ROW) * (_M) + (_COL)])
	#define IDX_HMATRIX(_ROW, _COL, _R, _M) ((_ROW) * (_M) + (_COL))
#else
	#define HMATRIX(_ROW, _COL, _R, _M) (H[(_COL) * (_R) + (_ROW)])
	#define IDX_HMATRIX(_ROW, _COL, _R, _M) ((_COL) * (_R) + (_ROW))
#endif

#ifdef ROW_MAJOR_W
	#define WMATRIX(_ROW, _COL, _N, _R) (W[(_ROW) * (_R) + (_COL)])
	#define IDX_WMATRIX(_ROW, _COL, _N, _R) ((_ROW) * (_R) + (_COL))
#else
	#define WMATRIX(_ROW, _COL, _N, _R) (W[(_COL) * (_N) + (_ROW)])
	#define IDX_WMATRIX(_ROW, _COL, _N, _R) ((_COL) * (_N) + (_ROW))
#endif

//#define SW(_R, _C) sw[(_R)][(_C)]
#define SW(_R, _C) (sw[(_C)][(_R)])

#define SVH(_R, _C) svh[(_R)][(_C)]
//#define SVH(_R, _C) (svh[(_C)][(_R)])

//#define SH(_R, _C) sh[(_R)][(_C)]
#define SH(_R, _C) sh[(_C)][(_R)]

#define SVW(_R, _C) svw[(_R)][(_C)]
//#define SVW(_R, _C) svw[(_C)][(_R)]

KERNEL UpdateW_AD(cudafloat * W, cudafloat * H, cudafloat * V, cudafloat * WH, cudafloat * sumH, int n, int m, int r) {
	__shared__ cudafloat SH(32, 32);
	__shared__ cudafloat SVW(32, 32);

	int x = blockIdx.x * 32 + threadIdx.x;	
	int y = blockIdx.y * 32 + threadIdx.y;

	cudafloat sum1 = CUDA_VALUE(0.0);
	//cudafloat sumH1 = CUDA_VALUE(0.0);
	cudafloat sum2 = CUDA_VALUE(0.0);
	//cudafloat sumH2 = CUDA_VALUE(0.0);

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
			cudafloat h1 = SH(threadIdx.x, i);
			cudafloat h2 = SH(tx, i);

			//sumH1 += h1;
			//sumH2 += h2;

			cudafloat vw = SVW(threadIdx.y, i) - CUDA_VALUE(1.0);

			sum1 += h1 * vw;
			sum2 += h2 * vw;
		}
		__syncthreads();
	}

	if (y < n && x < r) {
		int idx = IDX_WMATRIX(y, x, n, r);
		cudafloat sumH1 = sumH[x];
		cudafloat v = W[idx] + (W[idx] / sumH1) * sum1;		
		if (v < CUDA_VALUE(0.0)) v = CUDA_VALUE(0.0);
		W[idx] = v;

		x += 16;
		if (x  < r) {
			idx = IDX_WMATRIX(y, x, n, r);
			cudafloat sumH2 = sumH[x];
			v = W[idx] + (W[idx] / sumH2) * sum2;
			if (v < CUDA_VALUE(0.0)) v = CUDA_VALUE(0.0);
			W[idx] = v;
		}
	}
}

KERNEL UpdateH_AD(cudafloat * H, cudafloat * W, cudafloat * V, cudafloat * WH, cudafloat * sumW, int n, int m, int r) {
	__shared__ cudafloat SW(32, 32);
	__shared__ cudafloat SVH(32, 32);

	int x = blockIdx.x * 32 + threadIdx.x;	
	int y = blockIdx.y * 32 + threadIdx.y;

	cudafloat sum1 = CUDA_VALUE(0.0);
	//cudafloat sumW1 = CUDA_VALUE(0.0);
	cudafloat sum2 = CUDA_VALUE(0.0);
	//cudafloat sumW2 = CUDA_VALUE(0.0);

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
			cudafloat w1 = SW(i, threadIdx.y);
			cudafloat w2 = SW(i, ty);

			//sumW1 += w1;
			//sumW2 += w2;

			cudafloat vw = SVH(i, threadIdx.x) - CUDA_VALUE(1.0);

			sum1 += w1 * vw;
			sum2 += w2 * vw;
		}
		__syncthreads();
	}

	if (y < r && x < m) {
		int idx = IDX_HMATRIX(y, x, r, m);
		cudafloat sumW1 = sumW[y];
		cudafloat v = H[idx] + (H[idx] / sumW1) * sum1;
		if (v < CUDA_VALUE(0.0)) v = CUDA_VALUE(0.0);
		H[idx] = v;

		y+=16;
		if (y < r) {
			idx = IDX_HMATRIX(y, x, r, m);
			cudafloat sumW2 = sumW[y];
			v = H[idx] + (H[idx] / sumW2) * sum2;
			if (v < CUDA_VALUE(0.0)) v = CUDA_VALUE(0.0);
			H[idx] = v;
		}
	}
}

//! @}

}