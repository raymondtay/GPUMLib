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

#ifndef GPUMLib_DeviceArray_h
#define GPUMLib_DeviceArray_h

#include <string.h>
#include "HostArray.h"

namespace GPUMLib {

//! \addtogroup memframework Host (CPU) and device (GPU) memory access framework
//! @{

//! Create an array of any type, on the device, that automatically manages the memory used to hold its elements
template <class Type> class DeviceArray : public BaseArray<Type> {
	private:
		void Alloc(int size) {
			assert(size >= 0);

			if (size > 0 && cudaMalloc((void **) &(this->arrayData), size * sizeof(Type)) == cudaSuccess) {
				this->size = size;
			} else {
				this->Init();
			}
		}

	public:
		void Dispose() {
			if (this->size > 0) cudaFree(this->arrayData);
			this->Init();
		}

		//! Constructs an array with no elements
		DeviceArray() {}

		//! Constructs an array with size elements
		//! \param size number of elements of the array
		explicit DeviceArray(int size) {
			Alloc(size);
		}

		//! Constructs a device array with the same elements as an host array
		//! \param originalArray host array from where to copy the elements
		DeviceArray(const HostArray<Type> & originalArray) {
			Alloc(originalArray.Length());
			if (this->size > 0) cudaMemcpy(this->arrayData, originalArray.Pointer(), this->size * sizeof(Type), cudaMemcpyHostToDevice);
		}

		//! Constructs a device array with the same elements as another device array
		//! \param originalArray array from where to copy the elements
		DeviceArray(const DeviceArray<Type> & originalArray) {
			Alloc(originalArray.Length());
			if (this->size > 0) cudaMemcpy(this->arrayData, originalArray.arrayData, this->size * sizeof(Type), cudaMemcpyDeviceToDevice);
		}

		//! Constructs a device array with the same elements as those in an host array
		//! \param originalArray array data from where to copy the elements
		//! \param size number of elements to copy
		DeviceArray(const Type * originalArray, int size) {
			Alloc(size);
			if (this->size > 0) cudaMemcpy(this->arrayData, originalArray, size * sizeof(Type), cudaMemcpyHostToDevice);
		}

		#ifdef Cx11
		//! Constructs a device array using the elements of a temporary device array (rvalue)
		//! \param temporaryArray temporary array containing the elements
		DeviceArray(DeviceArray<Type> && temporaryArray) {
			MoveFrom(temporaryArray);
		}
		#endif

		//! Transforms this array into an array with the same data as an host array
		//! \param originalArray array from where to copy the elements
		//! \return a reference to this array
		DeviceArray<Type> & operator = (const HostArray<Type> & originalArray) {
			int newSize = originalArray.Length();
		
			this->ResizeWithoutPreservingData(newSize);
			if (this->size > 0) cudaMemcpy(this->arrayData, originalArray.Pointer(), this->size * sizeof(Type), cudaMemcpyHostToDevice);
			
			return *this;
		}

		//! Transforms this array into an array identical to another array
		//! \param originalArray array from where to copy the elements
		//! \return a reference to this array
		DeviceArray<Type> & operator = (const DeviceArray<Type> & originalArray) {
			int newSize = originalArray.Length();
		
			this->ResizeWithoutPreservingData(newSize);
			if (this->size > 0) cudaMemcpy(this->arrayData, originalArray.arrayData, this->size * sizeof(Type), cudaMemcpyDeviceToDevice);
			
			return *this;
		}

		#ifdef Cx11
		//! Replaces the elements of this device array by the elements of a temporary device array (rvalue)
		//! \param temporaryArray temporary array containing the elements
		//! \return a reference to this array
		DeviceArray<Type> & operator = (const DeviceArray<Type> && temporaryArray) {
			if (this != &temporaryArray) {
				Dispose();
				this->MoveFrom(temporaryArray);
			}
			
			return *this;
		}
		#endif

		//! Releases its own resources (elements) and obtains ownership of another array resources. 
		//! The other array will no longer have any elements. 
		//! In other words, it moves the elements from one device array to another.
		//! \param other array containing the elements to be moved.
		void TransferOwnerShipFrom(DeviceArray<Type> & other) {
			if (this != &other) {
				Dispose();
				this->size = other.size;
				this->arrayData = other.arrayData;
				other.Init();
			}
		}

		//! Destructor
		~DeviceArray() {
			Dispose();
		}
};

//! @}

}

#endif
