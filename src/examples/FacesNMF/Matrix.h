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

#ifndef Matrix_h
#define Matrix_h

#include "../../memory/HostMatrix.h"

using namespace GPUMLib;

// Create a matrix of any numeric type, on the host, that automaticaly manages the memory used to hold its elements
// warning Type must be a numeric type (e.g. int, float, double, ...).
template <class Type> class Matrix : public HostMatrix<Type> {
	public :
		Matrix() {}

		Matrix(int rows, int columns, StoringOrder storingOrder = RowMajor) : HostMatrix<Type>(rows, columns, storingOrder) {}

		Matrix(HostMatrix<Type> & other) : HostMatrix<Type>(other) {}

		Matrix<Type> & operator = (Matrix<Type> & other) {
			*((HostMatrix<Type> *) this) = (other);
			return *this;
		}

		Matrix<Type> & operator = (HostMatrix<Type> & other) {
			*((HostMatrix<Type> *) this) = (other);
			return *this;
		}

		static void Multiply(Matrix<Type> & A, Matrix<Type> & B, Matrix<Type> & C) {
			assert(C.rows == A.rows && C.columns == B.columns && A.columns == B.rows);

			for (int r = 0; r < C.rows; r++) {
				for (int c = 0; c < C.columns; c++) {
					Type value = 0; // implicit cast
					for(int l = 0; l < A.columns; l++) {
						value += A(r, l) * B(l, c);
					}

					C(r, c) = value;
				}
			}
		}

		void MultiplyBySelfTranspose(Matrix<Type> & C) {
			assert(C.rows == rows && C.columns == rows);
			
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < rows; c++) {
					Type value = 0; // implicit cast
					for(int l = 0; l < columns; l++) {
						value += operator ()(r, l) * operator ()(c, l);
					}

					C(r, c) = value;
				}
			}
		}

		Matrix<Type> operator * (Matrix<Type> & other) {
			Matrix<Type> result(rows, other.columns);
			Matrix<Type>::Multiply(*this, other, result);

			return result;
		}
};

#endif