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

#ifndef GPUMLib_reduction_h
#define GPUMLib_reduction_h

#include <cuda_runtime.h>
#include <cmath>

#include "../common/CudaDefinitions.h"
#include "../common/Utilities.h"
#include "../memory/CudaArray.h"
#include "../memory/DeviceMatrix.h"
#include "../memory/DeviceAccessibleVariable.h"

using namespace std;

namespace GPUMLib {

//! \addtogroup reduction Reduction framework
//! @{

//! Kernel to sum an array. For small arrays use KernelSumSmallArray instead.
//! \param[in] stream CUDA stream
//! \param[in] blocks Number of thread blocks 
//! \param[in] blockSize Block size (number of threads per block)
//! \param[in] inputs Values to be summed
//! \param[out] outputs Array that will contain the partial sums of each block
//! \param[in] numInputs Number of inputs
//! \sa KernelSumSmallArray, SIZE_SMALL_CUDA_VECTOR
void KernelSum(cudaStream_t stream, int blocks, int blockSize, cudafloat * inputs, cudafloat * outputs, int numInputs);

//! Kernel to sum a small array, multiply the result by a given factor and place the result in the output.
//! \param[in] stream CUDA stream
//! \param[in] blockSize Block size (number of threads per block)
//! \param[in] inputs Values to be summed
//! \param[out] output Pointer to the location that will contain the sum output
//! \param[in] numInputs Number of inputs
//! \param[in] multiplyFactor Multiply factor (optional, by default 1.0)
//! \sa KernelSum, SIZE_SMALL_CUDA_VECTOR
void KernelSumSmallArray(cudaStream_t stream, int blockSize, cudafloat * inputs, cudafloat * output, int numInputs, cudafloat multiplyFactor);

//! Kernel to compute the minimum of an array. 
//! \param[in] stream CUDA stream
//! \param[in] blocks Number of thread blocks
//! \param[in] blockSize Block size (number of threads per block)
//! \param[in] inputs input array
//! \param[out] output Pointer to the location that will contain the minimum
//! \param[in] numInputs Number of inputs
void KernelMin(cudaStream_t stream, int blocks, int blockSize, cudafloat * inputs, cudafloat * output, int numInputs);

//! Kernel to compute the minimum of an array and its index within the array. 
//! \param[in] stream CUDA stream
//! \param[in] blocks Number of thread blocks
//! \param[in] blockSize Block size (number of threads per block)
//! \param[in] inputs input array
//! \param[out] output Pointer to the location that will contain the minimum
//! \param[out] minIndexes Pointer to the location that will contain the index of one of the minimums
//! \param[in] numInputs Number of inputs
//! \param[in] indexes Buffer used to tempory store the indexes. Must have the same size of the inputs array.
void KernelMinIndexes(cudaStream_t stream, int blocks, int blockSize, cudafloat * inputs, cudafloat * output, int * minIndexes, int numInputs, int * indexes);

//! Kernel to compute the maximum of an array. 
//! \param[in] stream CUDA stream
//! \param[in] blocks Number of thread blocks
//! \param[in] blockSize Block size (number of threads per block)
//! \param[in] inputs input array
//! \param[out] output Pointer to the location that will contain the maximum
//! \param[in] numInputs Number of inputs
void KernelMax(cudaStream_t stream, int blocks, int blockSize, cudafloat * inputs, cudafloat * output, int numInputs);

//! Kernel to compute the maximum of an array and its index within the array. 
//! \param[in] stream CUDA stream
//! \param[in] blocks Number of thread blocks
//! \param[in] blockSize Block size (number of threads per block)
//! \param[in] inputs input array
//! \param[out] output Pointer to the location that will contain the maximum
//! \param[out] maxIndexes Pointer to the location that will contain the index of one of the maximums
//! \param[in] numInputs Number of inputs
//! \param[in] indexes Buffer used to tempory store the indexes. Must have the same size of the inputs array.
void KernelMaxIndexes(cudaStream_t stream, int blocks, int blockSize, cudafloat * inputs, cudafloat * output, int * maxIndexes, int numInputs, int * indexes);

//! Provides reduction functions (Sum, Average, Max, Min, ...).
class Reduction {
	private:
		void static Sum(cudafloat * inputs, cudafloat * output, int numInputs, cudafloat multiplyFactor, cudaStream_t stream);

		void static MinIndex(cudafloat * inputs, cudafloat * output, int * minIndex, int numInputs, cudaStream_t stream);
		void static Min(cudafloat * inputs, cudafloat * output, int numInputs, cudaStream_t stream);

		void static Max(cudafloat * inputs, cudafloat * output, int numInputs, cudaStream_t stream);
		void static MaxIndex(cudafloat * inputs, cudafloat * output, int * minIndex, int numInputs, cudaStream_t stream);

	public:
		//! Temporary buffer used for the reduction tasks. Programmers may take advantage of it for other tasks (hence, it is declared as public).
		static DeviceArray<cudafloat> temporaryBuffer;

		//! Sums all the elements of an input array, multiplies the sum by a given factor and places the result in the output
		//! \param[in] inputs Values to be summed
		//! \param[out] output Pointer to the memory address that will contain the sum output
		//! \param[in] multiplyFactor Multiply factor (optional, by default 1.0)
		//! \param[in] stream CUDA stream (optional)
		void static Sum(DeviceArray<cudafloat> & inputs, cudafloat * output, cudafloat multiplyFactor = CUDA_VALUE(1.0), cudaStream_t stream = nullptr) {
			Sum(inputs.Pointer(), output, inputs.Length(), multiplyFactor, stream);
		}

		//! Sums all the elements of an input array, multiplies the sum by a given factor and places the result in the output
		//! \param[in] inputs Values to be summed
		//! \param[out] output Array that will contain the sum output (in position 0)
		//! \param[in] multiplyFactor Multiply factor (optional, by default 1.0)
		//! \param[in] stream CUDA stream (optional)
		void static Sum(DeviceArray<cudafloat> & inputs, DeviceArray<cudafloat> & output, cudafloat multiplyFactor = CUDA_VALUE(1.0), cudaStream_t stream = nullptr) {
			Sum(inputs.Pointer(), output.Pointer(), inputs.Length(), multiplyFactor, stream);
		}

		//! Averages the elements of an input array, placing the result in the output
		//! \param[in] inputs input array for which we want to compute the average
		//! \param[out] output Array that will contain the average (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static Average(DeviceArray<cudafloat> & inputs, DeviceArray<cudafloat> & output, cudaStream_t stream = nullptr) {
			double multiplyFactor = 1.0 / inputs.Length();
			Sum(inputs.Pointer(), output.Pointer(), inputs.Length(), (cudafloat) multiplyFactor, stream);
		}

		//! Computes the minimum of an input array, placing the result in the output
		//! \param[in] inputs input array for which we want to compute the minimum
		//! \param[out] output Array that will contain the minimum (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static Min(DeviceArray<cudafloat> & inputs, DeviceArray<cudafloat> & output, cudaStream_t stream = nullptr) {
			Min(inputs.Pointer(), output.Pointer(), inputs.Length(), stream);
		}

		//! Computes the minimum of an input matrix, placing the result in the output
		//! \param[in] inputs input matrix for which we want to compute the minimum
		//! \param[out] output Array that will contain the minimum (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static Min(DeviceMatrix<cudafloat> & inputs, DeviceArray<cudafloat> & output, cudaStream_t stream = nullptr) {
			Min(inputs.Pointer(), output.Pointer(), inputs.Elements(), stream);
		}

		//! Computes the minimum of an input array as well as its index within the array
		//! \param[in] inputs input array for which we want to compute the minimum
		//! \param[out] min Array that will contain the minimum (in position 0)
		//! \param[out] minIndex Array that will contain the index of the minimum within the array (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static MinIndex(DeviceArray<cudafloat> & inputs, DeviceArray<cudafloat> & min, DeviceArray<int> & minIndex, cudaStream_t stream = nullptr) {
			MinIndex(inputs.Pointer(), min.Pointer(), minIndex.Pointer(), inputs.Length(), stream);
		}

		//! Computes the minimum of an input matrix as well as its (1-D) index within the matrix
		//! \param[in] inputs input matrix for which we want to compute the minimum
		//! \param[out] min Array that will contain the minimum (in position 0)
		//! \param[out] minIndex Array that will contain the index of the minimum within the array (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static MinIndex(DeviceMatrix<cudafloat> & inputs, DeviceArray<cudafloat> & min, DeviceArray<int> & minIndex, cudaStream_t stream = nullptr) {
			MinIndex(inputs.Pointer(), min.Pointer(), minIndex.Pointer(), inputs.Elements(), stream);
		}

		//! Computes the maximum of an input array, placing the result in the output
		//! \param[in] inputs input array for which we want to compute the maximum
		//! \param[out] output Array that will contain the maximum (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static Max(DeviceArray<cudafloat> & inputs, DeviceArray<cudafloat> & output, cudaStream_t stream = nullptr) {
			Max(inputs.Pointer(), output.Pointer(), inputs.Length(), stream);
		}

		//! Computes the maximum of an input matrix, placing the result in the output
		//! \param[in] inputs input matrix for which we want to compute the maximum
		//! \param[out] output Array that will contain the maximum (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static Max(DeviceMatrix<cudafloat> & inputs, DeviceArray<cudafloat> & output, cudaStream_t stream = nullptr) {
			Max(inputs.Pointer(), output.Pointer(), inputs.Elements(), stream);
		}

		//! Computes the maximum of an input array as well as its index within the array
		//! \param[in] inputs input array for which we want to compute the minimum
		//! \param[out] max Array that will contain the minimum (in position 0)
		//! \param[out] maxIndex Array that will contain the index of the minimum within the array (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static MaxIndex(DeviceArray<cudafloat> & inputs, DeviceArray<cudafloat> & max, DeviceArray<int> & maxIndex, cudaStream_t stream = nullptr) {
			MaxIndex(inputs.Pointer(), max.Pointer(), maxIndex.Pointer(), inputs.Length(), stream);
		}

		//! Computes the maximum of an input matrix as well as its (1-D) index within the array
		//! \param[in] inputs input matrix for which we want to compute the minimum
		//! \param[out] max Array that will contain the minimum (in position 0)
		//! \param[out] maxIndex Array that will contain the index of the minimum within the array (in position 0)
		//! \param[in] stream CUDA stream (optional)
		void static MaxIndex(DeviceMatrix<cudafloat> & inputs, DeviceArray<cudafloat> & max, DeviceArray<int> & maxIndex, cudaStream_t stream = nullptr) {
			MaxIndex(inputs.Pointer(), max.Pointer(), maxIndex.Pointer(), inputs.Elements(), stream);
		}
};

//! @}

}

#endif