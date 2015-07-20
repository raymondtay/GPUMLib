/*
	Noel Lopes is an Assistant Professor at the Polytechnic Institute of Guarda, Portugal
	Copyright (C) 2009, 2010, 2011, 2012, 2013, 2014 Noel de Jesus Mendonça Lopes

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

#ifndef HostNMF_h
#define HostNMF_h

//#include "stdafx.h"
//#include "common/CudaDefinitions.h"
//#include "memory/DeviceMatrix.h"
#include "Matrix.h"

//#include <float.h>

typedef enum NMF_METHOD {
	 MULTIPLICATIVE_EUCLIDEAN,
	 MULTIPLICATIVE_DIVERGENCE,
	 ADDITIVE_EUCLIDEAN,
	 ADDITIVE_DIVERGENCE
};

class HostNMF {
	public:
		Matrix<cudafloat> V;
		Matrix<cudafloat> W;
		Matrix<cudafloat> H;
		Matrix<cudafloat> WH;

	private:
		Matrix<cudafloat> WtV;
		Matrix<cudafloat> WtW;
		Matrix<cudafloat> WtWH;
		Matrix<cudafloat> VHt;
		Matrix<cudafloat> WHHt;		

		int n;
		int m;
		int r;

		static void Randomize(Matrix<cudafloat> & m) {
			int ne = m.Elements();
			cudafloat * p = m.Pointer();

			for(int e = 0; e < ne; e++) p[e] = ((cudafloat)rand()) / RAND_MAX;
		}

		void DoIteration_ME(bool updateW) {
			// Calculate Wt
			W.ReplaceByTranspose();
			Matrix<cudafloat> & Wt = W;

			// Calculate WtV
			Matrix<cudafloat>::Multiply(Wt, V, WtV);

			// Calculate WtW
			Wt.MultiplyBySelfTranspose(WtW);

			// Calculate WtWH
			Matrix<cudafloat>::Multiply(WtW, H, WtWH);

			int hRows = H.Rows();
			int hCols = H.Columns();

			for (int r = 0; r < hRows; r++) {
				for (int c = 0; c < hCols; c++) {
					H(r, c) *= WtV(r, c) / (WtWH(r, c) + CUDA_VALUE(10e-9));
				}
			}

			Wt.ReplaceByTranspose();

			if (updateW) {
				// Calculate Ht
				H.ReplaceByTranspose();
				Matrix<cudafloat> & Ht = H;

				// Calculate VHt
				Matrix<cudafloat>::Multiply(V, Ht, VHt);

				// Calculate HHt
				Matrix<cudafloat> & HHt = WtW;
				Ht.ReplaceByTranspose();
				H.MultiplyBySelfTranspose(HHt);

				// Calculate WHHt				
				Matrix<cudafloat>::Multiply(W, HHt, WHHt);

				int wRows = W.Rows();
				int wCols = W.Columns();

				for (int r = 0; r < wRows; r++) {
					for (int c = 0; c < wCols; c++) {
						W(r, c) *= VHt(r, c) / (WHHt(r, c) + CUDA_VALUE(10e-9));
					}
				}
			}
		}

		void DoIteration_MD(bool updateW) {
			// Update H
			Matrix<cudafloat>::Multiply(W, H, WH);
			for(int a = 0; a < r; a++) {
				for(int u = 0; u < m; u++) {
					cudafloat sum = CUDA_VALUE(0.0);
					cudafloat sumW = CUDA_VALUE(0.0);
					for(int i = 0; i < n; i++) {
						cudafloat w = W(i, a);
						sum += w * V(i, u) / (WH(i, u) + CUDA_VALUE(10e-9));
						sumW += w;
					}

					H(a, u) *= sum / sumW;
				}
			}

			
			if (updateW) { // Update W
				Matrix<cudafloat>::Multiply(W, H, WH);
				for(int i = 0; i < n; i++) {
					for(int a = 0; a < r; a++) {
						cudafloat sum = CUDA_VALUE(0.0);
						cudafloat sumH = CUDA_VALUE(0.0);
						for(int u = 0; u < m; u++) {
							cudafloat h = H(a, u);
							sum += h * V(i, u) / (WH(i, u) + CUDA_VALUE(10e-9));
							sumH += h;
						}

						W(i, a) *= sum / sumH;
					}
				}
			}
		}

		void DoIteration_AE(bool updateW) {
			// Update H
			W.ReplaceByTranspose();
			Matrix<cudafloat>::Multiply(W, V, WtV);
			W.MultiplyBySelfTranspose(WtW);
			Matrix<cudafloat>::Multiply(WtW, H, WtWH);
			W.ReplaceByTranspose();

			for (int a = 0; a < r; a++) {
				for (int u = 0; u < m; u++) {
					cudafloat h = H(a, u);
					cudafloat wtwh = WtWH(a, u);
					cudafloat lr =  h / wtwh;
					
					h += lr * (WtV(a, u) - wtwh);
					H(a, u) = (h < CUDA_VALUE(0.0)) ? CUDA_VALUE(0.0) : h;
				}
			}

			if (updateW) { // Update W
				H.ReplaceByTranspose();
				Matrix<cudafloat>::Multiply(V, H, VHt);
				H.ReplaceByTranspose();
				H.MultiplyBySelfTranspose(WtW);
				Matrix<cudafloat> & HHt = WtW;
				Matrix<cudafloat>::Multiply(W, HHt, WHHt);

				for (int i = 0; i < n; i++) {
					for (int a = 0; a < r; a++) {
						cudafloat w = W(i, a);
						cudafloat whht = WHHt(i, a);
						cudafloat lr =  w / whht;
					
						w += lr * (VHt(i, a) - whht);
						W(i, a) = (w < CUDA_VALUE(0.0)) ? CUDA_VALUE(0.0) : w;
					}
				}
			}
		}

		void DoIteration_AD(bool updateW) {
			// Update H
			Matrix<cudafloat>::Multiply(W, H, WH);
			for(int a = 0; a < r; a++) {
				for(int u = 0; u < m; u++) {
					cudafloat sum = CUDA_VALUE(0.0);
					cudafloat sumW = CUDA_VALUE(0.0);
					for(int i = 0; i < n; i++) {
						cudafloat w = W(i, a);
						sum += w * V(i, u) / (WH(i, u) + CUDA_VALUE(10e-9)) - w;
						sumW += w;
					}

					cudafloat h = H(a, u);
					cudafloat lr = h / sumW;
					
					h += lr * sum;
					H(a, u) = (h < CUDA_VALUE(0.0)) ? CUDA_VALUE(0.0) : h;
				}
			}

			if (updateW) { // Update W
				Matrix<cudafloat>::Multiply(W, H, WH);
				for(int i = 0; i < n; i++) {
					for(int a = 0; a < r; a++) {
						cudafloat sum = CUDA_VALUE(0.0);
						cudafloat sumH = CUDA_VALUE(0.0);
						for(int u = 0; u < m; u++) {
							cudafloat h = H(a, u);
							sum += h * V(i, u) / (WH(i, u) + CUDA_VALUE(10e-9)) - h;
							sumH += h;
						}

						cudafloat w = W(i, a);
						cudafloat lr = w / sumH;
						
						w += lr * sum;
						W(i, a) = (w < CUDA_VALUE(0.0)) ? CUDA_VALUE(0.0) : w;
					}
				}
			}
		}

	public:
		HostNMF(HostNMF * NMFtrainMatrices, int m) : V(NMFtrainMatrices->n, m, ColumnMajor), W(NMFtrainMatrices->W), H(NMFtrainMatrices->r, m, ColumnMajor), WH(NMFtrainMatrices->n, m) {
			this->n = NMFtrainMatrices->n;
			this->m = m;
			this->r = NMFtrainMatrices->r;

			WtV.ResizeWithoutPreservingData(r, m);
			WtW.ResizeWithoutPreservingData(r, r);
			WtWH.ResizeWithoutPreservingData(r, m);
			//VHt.ResizeWithoutPreservingData(n, r);
			//WHHt.ResizeWithoutPreservingData(n, r);

			Randomize(H);
		}

		HostNMF(int n, int m, int r) : V(n, m, ColumnMajor), W(n, r, ColumnMajor), H(r, m, ColumnMajor), WtV(r, m), WtW(r, r), WtWH(r, m), VHt(n, r), WHHt(n, r), WH(n, m) {
			this->n = n;
			this->m = m;
			this->r = r;
			Randomize();
		}

		void SetR(int r) {
			if (this->r != r) {
				this->r = r;

				W.ResizeWithoutPreservingData(n, r);
				H.ResizeWithoutPreservingData(r, m);				
				WtV.ResizeWithoutPreservingData(r, m);
				WtW.ResizeWithoutPreservingData(r, r);
				WtWH.ResizeWithoutPreservingData(r, m);
				VHt.ResizeWithoutPreservingData(n, r);
				WHHt.ResizeWithoutPreservingData(n, r);

				Randomize();
			}
		}

		void Randomize() {
			Randomize(W);
			Randomize(H);
		}

		void DoIteration(NMF_METHOD m, bool updateW) {
			switch (m) {
				case MULTIPLICATIVE_EUCLIDEAN:
					DoIteration_ME(updateW);
					break;
				case MULTIPLICATIVE_DIVERGENCE:
					DoIteration_MD(updateW);
					break;
				case ADDITIVE_EUCLIDEAN:
					DoIteration_AE(updateW);
					break;
				case ADDITIVE_DIVERGENCE:
					DoIteration_AD(updateW);
					break;
			}
		}
};

#endif