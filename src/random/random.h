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

#ifndef GPUMLib_Random_h
#define GPUMLib_Random_h

#include <curand.h>
#include "../memory/DeviceArray.h"

namespace GPUMLib {

//! \addtogroup random Random Generator
//! @{

//! Class for generating random values on the device. Uses the CURAND library.
class Random {
	private:
		static curandGenerator_t randomGenerator;
		static curandRngType_t randomGeneratorType;

		static curandGenerator_t RandomGenerator();

	public:
		//! Destroys the existing random generator and frees the memory associated with its state.
		static void CleanUp();

		//! Set the seed value of the pseudorandom number generator.
		//! \param seed the new seed to be used.
		//! \param generatorType Generator type. For more information see the CURAND documentation. Valid values include: CURAND_RNG_PSEUDO_DEFAULT, CURAND_RNG_PSEUDO_XORWOW, CURAND_RNG_QUASI_DEFAULT, CURAND_RNG_QUASI_SOBOL32, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32, CURAND_RNG_QUASI_SOBOL64, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64
		static void SetSeed(unsigned long long seed, curandRngType_t generatorType = CURAND_RNG_PSEUDO_DEFAULT);

		//! Fills a device array with random numbers between 0 and 1.
		//! \param a array to fill
		static void Fill(DeviceArray<float> & a);
};

//! @}

}

#endif