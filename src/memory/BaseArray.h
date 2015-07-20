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

#ifndef GPUMLib_BaseArray_h
#define GPUMLib_BaseArray_h

#include "../common/config.h"

namespace GPUMLib {

//! \addtogroup memframework Host (CPU) and device (GPU) memory access framework
//! @{

template <class Type> class CudaArray;

//! Base class for HostArray and DeviceArray classes (Array base class)
template <class Type> class BaseArray {
	friend class CudaArray<Type>;

	protected:
		Type * arrayData;
		int size;

		void Init() {
			arrayData = nullptr;
			size = 0;
		}

		BaseArray() {
			Init();
		}

		#ifdef Cx11
		void MoveFrom(BaseArray<Type> & other) {
			size = other.size;
			arrayData = other.arrayData;

			other.Init();
		}
		#endif

		virtual void Alloc(int size) = 0;

	public:
		//! Disposes the array.
		virtual void Dispose() = 0;

		//! Gets the length of the array. You can use this function to check if the array was effectively allocated.
		//! \return the number of elements of the array
		int Length() const {
			return size;
		}

		//! Gets a pointer to the array data
		//! \attention Use with caution
		//! \return a pointer to the array data
		Type * Pointer() const {
			return (size > 0) ? arrayData : nullptr;
		}

		//! Resizes the array without preserving its data
		//! \param size new size of the array
		//! \return the number of elements of the array after being resized.
		int ResizeWithoutPreservingData(int size) {
			if (size != this->size) {
				Dispose();
				Alloc(size);
			}

			return Length();
		}
};

//! @}

}

#endif