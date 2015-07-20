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

#include "NMFmultiplicativeDivergence.h"

namespace GPUMLib {

//! \addtogroup nmf Non-negative Matrix Factorization classes
//! @{

void NMF_MultiplicativeDivergence::DoIteration(bool updateW) {
	int n = V.Rows();
	int m = V.Columns();
	int r = W.Columns();
	
	// Update H
	DeviceMatrix<cudafloat>::Multiply(W, H, WH);

	DetermineQualityImprovement(false);

	KernelSumW(NumberThreadsPerBlockThatBestFit(n), W.Pointer(), n, r, sum.Pointer());

	UpdateH_MD<<<gh, bh>>>(H.Pointer(), W.Pointer(), V.Pointer(), WH.Pointer(), sum.Pointer(), n, m, r);

	if (!updateW) return;

	// Update W
	DeviceMatrix<cudafloat>::Multiply(W, H, WH);
	KernelSumH(NumberThreadsPerBlockThatBestFit(m), H.Pointer(), r, m, sum.Pointer());

	UpdateW_MD<<<gw, bw>>>(W.Pointer(), H.Pointer(), V.Pointer(), WH.Pointer(), sum.Pointer(), n, m, r);
}

//! @}

}