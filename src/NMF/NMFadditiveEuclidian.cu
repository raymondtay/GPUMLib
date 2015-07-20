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

#include "NMFadditiveEuclidian.h"
#include "../common/Utilities.h"

namespace GPUMLib {

//! \addtogroup nmf Non-negative Matrix Factorization classes
//! @{

// V (n x m) | W (n x r) | H (r x m)
void NMF_AdditiveEuclidian::DoIteration(bool updateW) {
	DetermineQualityImprovement(true);

	// Update H
	W.ReplaceByTranspose();
	DeviceMatrix<cudafloat>::Multiply(W, V, deltaH);
	W.MultiplyBySelfTranspose(aux);
	//DeviceMatrix<cudafloat>::Multiply(aux, H, deltaH, CUDA_VALUE(-1.0), CUDA_VALUE(1.0));
	DeviceMatrix<cudafloat>::Multiply(aux, H, deltaH2);
	W.ReplaceByTranspose();
	//UpdateMatrixNMFadditive<<<NumberBlocks(H.Elements(), SIZE_BLOCKS_NMF), SIZE_BLOCKS_NMF>>>(H.Pointer(), deltaH.Pointer(), CUDA_VALUE(0.001), H.Elements());
	UpdateMatrix_AE<<<NumberBlocks(H.Elements(), SIZE_BLOCKS_NMF), SIZE_BLOCKS_NMF>>>(H.Pointer(), deltaH.Pointer(), deltaH2.Pointer(), H.Elements());

	if (!updateW) return;

	// Update W
	H.ReplaceByTranspose();
	DeviceMatrix<cudafloat>::Multiply(V, H, deltaW);
	H.ReplaceByTranspose();
	H.MultiplyBySelfTranspose(aux);
	//DeviceMatrix<cudafloat>::Multiply(W, aux, deltaW, CUDA_VALUE(-1.0), CUDA_VALUE(1.0));
	DeviceMatrix<cudafloat>::Multiply(W, aux, deltaW2);
	//UpdateMatrixNMFadditive<<<NumberBlocks(W.Elements(), SIZE_BLOCKS_NMF), SIZE_BLOCKS_NMF>>>(W.Pointer(), deltaW.Pointer(), CUDA_VALUE(0.001), W.Elements());
	UpdateMatrix_AE<<<NumberBlocks(W.Elements(), SIZE_BLOCKS_NMF), SIZE_BLOCKS_NMF>>>(W.Pointer(), deltaW.Pointer(), deltaW2.Pointer(), W.Elements());
}

//! @}

}