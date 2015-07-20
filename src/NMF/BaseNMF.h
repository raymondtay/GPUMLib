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

#ifndef GPUMLib_BaseNMF_h
#define GPUMLib_BaseNMF_h

#include "../common/CudaDefinitions.h"
#include "../common/CudaStreams.h"
#include "../common/Utilities.h"
#include "../memory/DeviceArray.h"
#include "../memory/DeviceMatrix.h"
#include "../memory/DeviceAccessibleVariable.h"
#include "NMFkernels.h"

#define SIZE_BLOCKS_NMF (512)

namespace GPUMLib {

//! \addtogroup nmf Non-negative Matrix Factorization classes
//! @{

//! Base class for all Non-Negative Matrix Factorization classes
class NMF {
	private:
		CudaStream streamQuality;
		
		int timesQualityWasObtained;
		cudafloat lastValueQuality;
		DeviceAccessibleVariable<cudafloat> vquality;
		cudafloat deltaImprovement;
		DeviceArray<cudafloat> d_quality;


		static void Randomize(HostMatrix<cudafloat> & m) {
			int ne = m.Elements();
			cudafloat * p = m.Pointer();

			for(int e = 0; e < ne; e++) p[e] = ((cudafloat) rand()) / RAND_MAX + SMALL_VALUE_TO_ADD_DENOMINATOR;
		}

	protected:
		DeviceMatrix<cudafloat> V;
		DeviceMatrix<cudafloat> W;
		DeviceMatrix<cudafloat> H;
		DeviceMatrix<cudafloat> WH;


		void DetermineQualityImprovement(bool calculateWH) {
			if (cudaStreamQuery(streamQuality) == cudaSuccess) {
				cudafloat quality = vquality.Value();
		
				if (timesQualityWasObtained++ > 2) {
					deltaImprovement = quality - lastValueQuality; 
				} else {
					timesQualityWasObtained++;
				}

				lastValueQuality = quality;

				if (calculateWH) DeviceMatrix<cudafloat>::Multiply(W, H, WH);

				int elements = WH.Elements();
				KernelNMFquality(NumberThreadsPerBlockThatBestFit(elements), V.Pointer(), WH.Pointer(), elements, d_quality.Pointer());

				vquality.UpdateValueAsync(d_quality.Pointer(), streamQuality);
			}
		}
		
		NMF(HostMatrix<cudafloat> & v, int r) : V(v), WH(v.Rows(), v.Columns(), ColumnMajor), d_quality(1) {
			assert(!v.IsRowMajor());

			timesQualityWasObtained = 0;
			deltaImprovement = CUDA_VALUE(1.0);
			lastValueQuality = CUDA_VALUE(-1.0);
			vquality.Value() = CUDA_VALUE(0.0);

			HostMatrix<cudafloat> aux(v.Rows(), r, ColumnMajor);
			Randomize(aux);
			W = aux;

			aux.ResizeWithoutPreservingData(r, v.Columns());
			Randomize(aux);
			H = aux;
		}

		NMF(HostMatrix<cudafloat> & v, HostMatrix<cudafloat> & w, HostMatrix<cudafloat> & h) : V(v), W(w), H(h), WH(v.Rows(), v.Columns(), ColumnMajor), d_quality(1) {
			assert(!v.IsRowMajor());
			assert(!w.IsRowMajor());
			assert(!h.IsRowMajor());

			assert(v.Rows() == w.Rows() && v.Columns() == h.Columns());
			assert(w.Columns() == h.Rows());

			timesQualityWasObtained = 0;
			deltaImprovement = CUDA_VALUE(1.0);
			lastValueQuality = CUDA_VALUE(-1.0);
			vquality.Value() = CUDA_VALUE(0.0);
		}

	public:
		//! Gets the W matrix
		//! \return the W matrix
		HostMatrix<cudafloat> GetW() {
			return HostMatrix<cudafloat>(W);
		}

		//! Gets the H matrix
		//! \return the H matrix
		HostMatrix<cudafloat> GetH() {
			return HostMatrix<cudafloat>(H);
		}

		//! Gets the approximation, given by WH, to the matrix V
		//! \return the approximation, given by WH, to the matrix V
		HostMatrix<cudafloat> GetWH() {
			DeviceMatrix<cudafloat>::Multiply(W, H, WH);

			return HostMatrix<cudafloat>(WH);
		}

		//! Do an algorithm iteration. Adjusts W and H matrices.
		//! \param updateW Indicates if the matrix W is updated (by default yes). 
		virtual void DoIteration(bool updateW = true) = 0;

		//! Gets the quality improvement caused by the last iteration
		cudafloat QualityImprovement() const {
			return deltaImprovement;
		}
};

//! @}

}

#endif