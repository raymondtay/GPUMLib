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

#ifndef GPUMLib_Utilities_h
#define GPUMLib_Utilities_h

#include "CudaDefinitions.h"
#include "../memory/HostArray.h"

namespace GPUMLib {

//! \addtogroup commonframework Common framework
//! @{

//! Finds the number of threads (multiple of 2) per block that either is greater that the number of threads needed or identical to the maximum number of threads per block.
//! \param threads Number of threads.
//! \param maxThreadsPerBlock Maximum number of threads.
//! \return The number of threads (multiple of 2) per block that either is greater that the number of threads needed or identical to the maximum number of threads per block.
//! \sa MAX_THREADS_PER_BLOCK, NumberBlocks
inline int NumberThreadsPerBlockThatBestFit(int threads, int maxThreadsPerBlock = MAX_THREADS_PER_BLOCK) {
	int nt = 1;
	while(nt < threads && nt < maxThreadsPerBlock) nt <<= 1;

	return nt;
}

//! Finds the number of blocks needed to execute the number of threads specified, given a block size.
//! \param threads Number of threads.
//! \param blockSize Block size.
//! \return The number of blocks needed to execute the number of threads specified.
//! \sa NumberThreadsPerBlockThatBestFit, MAX_THREADS_PER_BLOCK
inline int NumberBlocks(int threads, int blockSize) {
	int nb = threads / blockSize;

	if (threads % blockSize != 0) nb++;

	return nb;
}

//! @}

}

#endif