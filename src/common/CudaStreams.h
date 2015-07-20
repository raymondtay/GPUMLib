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

#ifndef GPUMLib_CudaStreams_h
#define GPUMLib_CudaStreams_h

#include <cuda_runtime.h>

namespace GPUMLib {

//! \addtogroup commonframework Common framework
//! @{

//! Represents a CUDA stream
class CudaStream {
	private:
		cudaStream_t stream;

	public:
		//! Constructor
		CudaStream() {
			cudaStreamCreate(&stream);
		}

		//! Destructor
		~CudaStream() {
			cudaStreamDestroy(stream);
		}

		//! \return Returns the stream as a \b cudaStream_t type
		operator cudaStream_t () {
			return stream;
		}
};

//! @}

}

#endif