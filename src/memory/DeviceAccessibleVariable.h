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

#ifndef GPUMLib_DeviceAccessibleVariable_h
#define GPUMLib_DeviceAccessibleVariable_h

namespace GPUMLib {

//! \addtogroup memframework Host (CPU) and device (GPU) memory access framework
//! @{

//! Represents a variable residing in memory that is page-locked and accessible to the device.
template <class Type> class DeviceAccessibleVariable {
	private:
		Type * value;

	public:
		//! Constructor
		DeviceAccessibleVariable() {
			cudaMallocHost((void**) &value, sizeof(Type));
		}

		//! Constructor
		//! \param initialValue Initial value
		DeviceAccessibleVariable(const Type initialValue) {
			cudaMallocHost((void**) &value, sizeof(Type));
			*value = initialValue;
		}

		//! Destructor
		~DeviceAccessibleVariable() {
			cudaFreeHost(value);
		}

		//! Gets a reference to the variable
		//! \return a reference to the variable
		Type & Value() {
			return *value;
		}

		//! Gets a pointer to the variable
		//! \return a pointer to the variable
		Type * Pointer() {
			return value;
		}

		//! Updates the variable value from a device memory variable
		//! \param deviceValue a pointer to the variable on the device
		void UpdateValue(Type * deviceValue) {
			cudaMemcpy(value, deviceValue, sizeof(Type), cudaMemcpyDeviceToHost);
		}

		//! Asynchronously updates the variable value from a device memory variable
		//! \param deviceValue a pointer to the variable on the device
		//! \param stream The CUDA stream used to transfer the data 
		void UpdateValueAsync(Type * deviceValue, cudaStream_t stream) {
			cudaMemcpyAsync(value, deviceValue, sizeof(Type), cudaMemcpyDeviceToHost, stream);
		}
};

//! @}

}

#endif