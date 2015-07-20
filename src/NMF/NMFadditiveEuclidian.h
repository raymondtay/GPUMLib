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

#ifndef GPUMLib_NMFadditiveEuclidian_h
#define GPUMLib_NMFadditiveEuclidian_h

#include "../memory/DeviceArray.h"
#include "BaseNMF.h"

namespace GPUMLib {

//! \addtogroup nmf Non-negative Matrix Factorization classes
//! @{

//! Represents a Non-Negative Matrix Factorization (NMF) algorithm that uses additive update rules and the Euclidean distance metric.
class NMF_AdditiveEuclidian : public NMF {
	private:
		DeviceMatrix<cudafloat> deltaH;
		DeviceMatrix<cudafloat> deltaH2;
		DeviceMatrix<cudafloat> deltaW;
		DeviceMatrix<cudafloat> deltaW2;
		DeviceMatrix<cudafloat> aux;

	public:
		//! Constructs a Non-negative Matrix Factorization (NMF) algorithm object that uses additive update rules and the Euclidean distance metric.
		//! Given a non-negative matrix V (n x m), the NMF algorithm will find non-negative matrix factors W (n x r) and H (r x m) such that V is approximately equal to WH.
		//! \param v n x m matrix (V) containing a set of multivariate n-dimensional data vectors. m is the number of examples in the dataset.
		//! \param r Determines the size of matrices W (n x r) and H (r x m).
		//! \attention Matrix V must be in column-major order
		NMF_AdditiveEuclidian(HostMatrix<cudafloat> & v, int r) : NMF(v, r), deltaH(r, v.Columns()), deltaW(v.Rows(), r), aux(r, r), deltaH2(r, v.Columns()), deltaW2(v.Rows(), r) {
		}

		//! Constructs a Non-negative Matrix Factorization (NMF) algorithm object that uses additive update rules and the Euclidean distance metric.
		//! Given a non-negative matrix V (n x m), the NMF algorithm will find non-negative matrix factors W (n x r) and H (r x m) such that V is approximately equal to WH.
		//! \param v n x m matrix (V) containing a set of multivariate n-dimensional data vectors. m is the number of examples in the dataset.
		//! \param w n x r matrix. 
		//! \param h r x m matrix.
		//! \attention Matrix V must be in column-major order
		NMF_AdditiveEuclidian(HostMatrix<cudafloat> & v, HostMatrix<cudafloat> & w, HostMatrix<cudafloat> & h) : NMF(v, w, h), deltaH(h.Rows(), v.Columns()), deltaW(v.Rows(), w.Columns()), aux(w.Columns(), w.Columns()), deltaH2(h.Rows(), v.Columns()), deltaW2(v.Rows(), w.Columns()) {
		}

		//! Do an algorithm iteration. Adjusts W and H matrices
		//! \param updateW Specifies if the matrix W is to be updated.
		void DoIteration(bool updateW = true);
};

//! @}

}

#endif