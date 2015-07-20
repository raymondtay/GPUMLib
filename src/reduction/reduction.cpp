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

#include "reduction.h"

namespace GPUMLib {

DeviceArray<cudafloat> Reduction::temporaryBuffer;

void Reduction::Sum(cudafloat * inputs, cudafloat * output, int numInputs, cudafloat multiplyFactor, cudaStream_t stream) {
	int blockSize = NumberThreadsPerBlockThatBestFit(numInputs, OPTIMAL_BLOCK_SIZE_REDUCTION);

	if (numInputs > SIZE_SMALL_CUDA_VECTOR) {
		int blocks = NumberBlocks(numInputs, blockSize);
		if (temporaryBuffer.Length() < blocks) temporaryBuffer.ResizeWithoutPreservingData(blocks);

		KernelSum(stream, blocks, blockSize, inputs, temporaryBuffer.Pointer(), numInputs);

		inputs = temporaryBuffer.Pointer();
		numInputs = blocks;

		blockSize = NumberThreadsPerBlockThatBestFit(numInputs);
	}
			
	KernelSumSmallArray(stream, blockSize, inputs, output, numInputs, multiplyFactor);
}

void Reduction::Min(cudafloat * inputs, cudafloat * output, int numInputs, cudaStream_t stream) {
	int blockSize = NumberThreadsPerBlockThatBestFit(numInputs, OPTIMAL_BLOCK_SIZE_REDUCTION);

	if (numInputs > SIZE_SMALL_CUDA_VECTOR) {
		int blocks = NumberBlocks(numInputs, blockSize);
		if (temporaryBuffer.Length() < blocks) temporaryBuffer.ResizeWithoutPreservingData(blocks);

		KernelMin(stream, blocks, blockSize, inputs, temporaryBuffer.Pointer(), numInputs);

		inputs = temporaryBuffer.Pointer();
		numInputs = blocks;

		blockSize = NumberThreadsPerBlockThatBestFit(numInputs);
	}

	KernelMin(stream, 1, blockSize, inputs, output, numInputs);
}

void Reduction::MinIndex(cudafloat * inputs, cudafloat * output, int * minIndex, int numInputs, cudaStream_t stream) {
	int blockSize = NumberThreadsPerBlockThatBestFit(numInputs, OPTIMAL_BLOCK_SIZE_REDUCTION);

	int * indexes = nullptr;

	if (numInputs > SIZE_SMALL_CUDA_VECTOR) {
		int blocks = NumberBlocks(numInputs, blockSize);

		int minSizeBuffer = blocks + (int) ceil(blocks * (sizeof(int) / (float) sizeof(float)));
		if (temporaryBuffer.Length() < minSizeBuffer) temporaryBuffer.ResizeWithoutPreservingData(minSizeBuffer);

		indexes = (int *)(temporaryBuffer.Pointer() + blocks);

		KernelMinIndexes(stream, blocks, blockSize, inputs, temporaryBuffer.Pointer(), indexes, numInputs, nullptr);

		inputs = temporaryBuffer.Pointer();
		numInputs = blocks;

		blockSize = NumberThreadsPerBlockThatBestFit(numInputs);
	}

	KernelMinIndexes(stream, 1, blockSize, inputs, output, minIndex, numInputs, indexes);
}

void Reduction::Max(cudafloat * inputs, cudafloat * output, int numInputs, cudaStream_t stream) {
	int blockSize = NumberThreadsPerBlockThatBestFit(numInputs, OPTIMAL_BLOCK_SIZE_REDUCTION);

	if (numInputs > SIZE_SMALL_CUDA_VECTOR) {
		int blocks = NumberBlocks(numInputs, blockSize);
		if (temporaryBuffer.Length() < blocks) temporaryBuffer.ResizeWithoutPreservingData(blocks);

		KernelMax(stream, blocks, blockSize, inputs, temporaryBuffer.Pointer(), numInputs);

		inputs = temporaryBuffer.Pointer();
		numInputs = blocks;

		blockSize = NumberThreadsPerBlockThatBestFit(numInputs);
	}

	KernelMax(stream, 1, blockSize, inputs, output, numInputs);
}

void Reduction::MaxIndex(cudafloat * inputs, cudafloat * output, int * maxIndex, int numInputs, cudaStream_t stream) {
	int blockSize = NumberThreadsPerBlockThatBestFit(numInputs, OPTIMAL_BLOCK_SIZE_REDUCTION);

	int * indexes = nullptr;

	if (numInputs > SIZE_SMALL_CUDA_VECTOR) {
		int blocks = NumberBlocks(numInputs, blockSize);

		int minSizeBuffer = blocks + (int) ceil(blocks * (sizeof(int) / (float) sizeof(float)));
		if (temporaryBuffer.Length() < minSizeBuffer) temporaryBuffer.ResizeWithoutPreservingData(minSizeBuffer);

		indexes = (int *)(temporaryBuffer.Pointer() + blocks);

		KernelMaxIndexes(stream, blocks, blockSize, inputs, temporaryBuffer.Pointer(), indexes, numInputs, nullptr);

		inputs = temporaryBuffer.Pointer();
		numInputs = blocks;

		blockSize = NumberThreadsPerBlockThatBestFit(numInputs);
	}

	KernelMaxIndexes(stream, 1, blockSize, inputs, output, maxIndex, numInputs, indexes);
}

}