/*
Joao Goncalves is a MSc Student at the University of Coimbra, Portugal
Copyright (C) 2012 Joao Goncalves

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

#ifndef SVM_H_
#define SVM_H_

#ifndef DEBUG
//! Comment or set this macro to zero to disable some runtime debugging info
#define DEBUG 1
#endif

#include "svm_kernel_type.h"

//GPUMLib stuff
#include "../common/CudaDefinitions.h"
#include "../common/Utilities.h"
#include "../memory/DeviceArray.h"
#include "../memory/DeviceMatrix.h"
#include "../memory/HostArray.h"
#include "../memory/HostMatrix.h"
#include <cuda.h>

#include "SVM_cuda_code.h"

namespace GPUMLib {

	//! \addtogroup smvdev Support Vector Machines class
	//! @{

	//! Represents an SVM which can be used to train and classify datasets on the GPU device
	class SVM {
	public:

		//! Constructor to represent a SVM on the GPU. Used as a placeholder to execute device code.
		SVM(){
			//do nothing
		}

		//! Gets the indices of the non zero alphas (Support Vectors)
		//! \param h_alphas A HostArray containing the alphas
		//! \param alpha_indices An array of integers to store the indices of the non zero alphas
		//! \param size The size of the input alphas array (actually, the amount of samples)
		//! \return The amount of non zero alphas (Support Vectors) found
		int getSupportVectorIndices(GPUMLib::HostArray<cudafloat> & h_alphas, int * alpha_indices, int size) {
			int non_zero_counter = 0;
			for (int i = 0; i < size; i++) {
				double alpha = h_alphas[i];
				if (alpha > 0.0) {
					alpha_indices[non_zero_counter] = i;
					non_zero_counter++;
				}
			}
			return non_zero_counter;
		}

		/**
		* Finds the minimum (first) position in the array where the target (int) occurs
		* @param array The HostArray where to search
		* @param array_length The element to search for
		* @param target The value to find in the given array
		* @return The position of the given target in the array (if found), -1 if not found
		*/
		int findMinimumPositionTarget_HostArray(GPUMLib::HostArray<int> & array, int array_length, int target) {
			for (int i = 0; i < array_length; i++) {
				int val = array[i];
				//cout << i << ":" << val << endl;
				if (val == target) {
					return i;
				}
			}
			return -1;
		}

		/**
		* Does the first reduction pass (using multiple blocks) to compute the hyperplane's offset (b) - basically a sum-reduce function

		* @param stream A cudaStream_t context to associate this execution with
		* @param blocks The number of CUDA blocks
		* @param blockSize The size of each CUDA block (must be a power of 2)
		* @param offsets The array with the offsets (the real output of the SVM classification without the sgn(...) operator) of each Support Vector
		* @param results The output array where to reduce (sum) the input offsets, where each index corresponds to a individual block
		* @param n_svs The size of the offsets array (number of SVs)
		**/
		void calculateB_1stPass(cudaStream_t stream,
			int blocks, int blockSize,
			cudafloat * offsets,
			cudafloat * results,
			int n_svs) {
				if (blockSize == 1024) {
					cuCalculateB_1stPass	<1024><<<blocks, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(offsets, results, n_svs);
				} else if(blockSize == 512) {
					cuCalculateB_1stPass<512><<<blocks, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(offsets, results, n_svs);
				} else if(blockSize == 256) {
					cuCalculateB_1stPass<256><<<blocks, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(offsets, results, n_svs);
				} else if(blockSize == 128) {
					cuCalculateB_1stPass<128><<<blocks, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(offsets, results, n_svs);
				} else if(blockSize == 64) {
					cuCalculateB_1stPass<64><<<blocks, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(offsets, results, n_svs);
				} else if(blockSize == 32) {
					cuCalculateB_1stPass<32><<<blocks, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(offsets, results, n_svs);
				} else if(blockSize == 16) {
					cuCalculateB_1stPass<16><<<blocks, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(offsets, results, n_svs);
				} else if(blockSize == 8) {
					cuCalculateB_1stPass<8><<<blocks, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(offsets, results, n_svs);
				} else if(blockSize == 4) {
					cuCalculateB_1stPass<4><<<blocks, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(offsets, results, n_svs);
				} else if(blockSize == 2) {
					cuCalculateB_1stPass<2><<<blocks, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(offsets, results, n_svs);
				} else if(blockSize == 1) {
					cuCalculateB_1stPass<1><<<blocks, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(offsets, results, n_svs);
				}
		}

		/**
		* Does the last reduction pass (using one block) to compute the hyperplane's offset (b) - basically a sum-reduce function
		* @param stream A cudaStream_t context to associate this execution with
		* @param blockSize The size of each CUDA block (must be a power of 2)
		* @param input_floats The array with the offsets (the real output of the SVM classification without the sgn(...) operator) of each Support Vector resulting from the first pass
		* @param input_size The size of the offsets array (number of blocks in the first passage)
		**/
		void calculateB_FinalPass(cudaStream_t stream, int blockSize, cudafloat * input_floats, int input_size) {
			switch (blockSize) {
	case 1024:
		cuCalculateB_FinalPass<1024><<<1, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(input_floats, input_size);
		break;
	case 512:
		cuCalculateB_FinalPass<512><<<1, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(input_floats, input_size);
		break;
	case 256:
		cuCalculateB_FinalPass<256><<<1, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(input_floats, input_size);
		break;
	case 128:
		cuCalculateB_FinalPass<128><<<1, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(input_floats, input_size);
		break;
	case 64:
		cuCalculateB_FinalPass<64><<<1, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(input_floats, input_size);;
		break;
	case 32:
		cuCalculateB_FinalPass<32><<<1, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(input_floats, input_size);
		break;
	case 16:
		cuCalculateB_FinalPass<16><<<1, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(input_floats, input_size);
		break;
	case 8:
		cuCalculateB_FinalPass<8><<<1, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(input_floats, input_size);
		break;
	case 4:
		cuCalculateB_FinalPass<4><<<1, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(input_floats, input_size);
		break;
	case 2:
		cuCalculateB_FinalPass<2><<<1, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(input_floats, input_size);
		break;
	case 1:
		cuCalculateB_FinalPass<1><<<1, blockSize, blockSize * (sizeof(cudafloat)), stream>>>(input_floats, input_size);
		break;
			}
		}

		/**
		* Does the first pass of the first order alphas search heuristic

		* @param stream A cudaStream_t context to associate this execution with
		* @param blocks The number of CUDA blocks
		* @param blockSize The size of each CUDA block (must be a power of 2)
		* @param y The array containing the classes, one for each sample
		* @param f The array of optimality conditions for each sample
		* @param alphas The array of alphas (lagrange multipliers)

		* @param minimuns The array where to store the minimum f value found in this CUDA block
		* @param min_indices The position of the minimum found in this search in the array f

		* @param maximuns The array where to store the maximum f value found in this CUDA block
		* @param max_indices The position of the maximum found in this search in the array f

		* @param input_size The size of the array f, which is equal to the amount of samples
		* @param constant_epsilon The epsilon tolerance used in the first order search heuristic
		* @param constant_c The penalization constant.
		**/
		void kernelFirstOrderHeuristic1stPass(cudaStream_t stream, int blocks, int blockSize,
			cudafloat * f, cudafloat * alphas,
			int * y, cudafloat * minimuns, int * min_indices, 
			cudafloat * maximuns, int * max_indices, int input_size,
			cudafloat constant_epsilon, cudafloat constant_c) {
				switch (blockSize) {
	case 1024:
		cuFirstOrderHeuristic1stPass<1024><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(f, alphas, y, minimuns, min_indices, maximuns, max_indices, input_size, constant_epsilon, constant_c);
		break;
	case 512:
		cuFirstOrderHeuristic1stPass<512><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(f, alphas, y, minimuns, min_indices, maximuns, max_indices, input_size, constant_epsilon, constant_c);
		break;
	case 256:
		cuFirstOrderHeuristic1stPass<256><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(f, alphas, y, minimuns, min_indices, maximuns, max_indices, input_size, constant_epsilon, constant_c);
		break;
	case 128:
		cuFirstOrderHeuristic1stPass<128><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(f, alphas, y, minimuns, min_indices, maximuns, max_indices, input_size, constant_epsilon, constant_c);
		break;
	case 64:
		cuFirstOrderHeuristic1stPass<64><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(f, alphas, y, minimuns, min_indices, maximuns, max_indices, input_size, constant_epsilon, constant_c);
		break;
	case 32:
		cuFirstOrderHeuristic1stPass<32><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(f, alphas, y, minimuns, min_indices, maximuns, max_indices, input_size, constant_epsilon, constant_c);
		break;
	case 16:
		cuFirstOrderHeuristic1stPass<16><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(f, alphas, y, minimuns, min_indices, maximuns, max_indices, input_size, constant_epsilon, constant_c);
		break;
	case 8:
		cuFirstOrderHeuristic1stPass<8><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(f, alphas, y, minimuns, min_indices, maximuns, max_indices, input_size, constant_epsilon, constant_c);
		break;
	case 4:
		cuFirstOrderHeuristic1stPass<4><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(f, alphas, y, minimuns, min_indices, maximuns, max_indices, input_size, constant_epsilon, constant_c);
		break;
	case 2:
		cuFirstOrderHeuristic1stPass<2><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(f, alphas, y, minimuns, min_indices, maximuns, max_indices, input_size, constant_epsilon, constant_c);
		break;
	case 1:
		cuFirstOrderHeuristic1stPass<1><<<blocks, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(f, alphas, y, minimuns, min_indices, maximuns, max_indices, input_size, constant_epsilon, constant_c);
		break;
				}
		}

		/**
		* Does the last pass of the first order alphas search heuristic. This function basically performs a simultaneous max/min reduction. The results are store in device's variables: d_b_high, d_i_high, d_b_low, d_i_low.

		* @param stream A cudaStream_t context to associate this execution with
		* @param blockSize The size of each CUDA block (must be a power of 2)

		* @param minimuns_input The array containing a list of values to search for a minimum
		* @param min_indices_input The position of the minimum found in this search in the optimality array

		* @param maximuns_input The array containing a list of values to search for a maximum
		* @param max_indices_input The position of the maximum found in this search in the optimality array

		* @param input_size The size of the array input arrays, which is equal to the amount blocks in the previous pass
		**/
		void kernelFirstOrderHeuristicFinalPass(cudaStream_t stream, int blockSize,
			cudafloat * minimuns_input, int * min_indices_input,
			cudafloat * maximuns_input,	int * max_indices_input,
			int input_size) {
				switch (blockSize) {
	case 1024:
		cuFirstOrderHeuristicFinalPass<1024><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(minimuns_input, min_indices_input, maximuns_input, max_indices_input, input_size);
		break;
	case 512:
		cuFirstOrderHeuristicFinalPass<512><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(minimuns_input, min_indices_input, maximuns_input, max_indices_input, input_size);
		break;
	case 256:
		cuFirstOrderHeuristicFinalPass<256><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(minimuns_input, min_indices_input, maximuns_input, max_indices_input, input_size);
		break;
	case 128:
		cuFirstOrderHeuristicFinalPass<128><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(minimuns_input, min_indices_input, maximuns_input, max_indices_input, input_size);
		break;
	case 64:
		cuFirstOrderHeuristicFinalPass<64><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(minimuns_input, min_indices_input, maximuns_input, max_indices_input, input_size);;
		break;
	case 32:
		cuFirstOrderHeuristicFinalPass<32><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(minimuns_input, min_indices_input, maximuns_input, max_indices_input, input_size);
		break;
	case 16:
		cuFirstOrderHeuristicFinalPass<16><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(minimuns_input, min_indices_input, maximuns_input, max_indices_input, input_size);
		break;
	case 8:
		cuFirstOrderHeuristicFinalPass<8><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(minimuns_input, min_indices_input, maximuns_input, max_indices_input, input_size);
		break;
	case 4:
		cuFirstOrderHeuristicFinalPass<4><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(minimuns_input, min_indices_input, maximuns_input, max_indices_input, input_size);
		break;
	case 2:
		cuFirstOrderHeuristicFinalPass<2><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(minimuns_input, min_indices_input, maximuns_input, max_indices_input, input_size);
		break;
	case 1:
		cuFirstOrderHeuristicFinalPass<1><<<1, blockSize, blockSize * (sizeof(cudafloat) + sizeof(int) + sizeof(cudafloat) + sizeof(int)), stream>>>(minimuns_input, min_indices_input, maximuns_input, max_indices_input, input_size);
		break;
				}
		}

		/**
		* Updates the alphas stored in the device's memory (old/new alphas for both low and high indices). Currently only using the negative C constant for both classes.

		* @param stream A cudaStream_t context to associate this execution with
		* @param kernel_type The svm_kernel_type (enum) of used kernel (gaussian, linear, ukf, etc.)
		* @param d_x The DeviceMatrix containing the attributes for each sample
		* @param d_alphas The DeviceArray containing the alpha associated with each training sample
		* @param d_y The DeviceArray containing the array of classes, one for each training sample
		* @param constant_c_negative The penalization constant associated with the negative class
		* @param constant_c_positive The penalization constant associated with the positive class
		* @param d_kernel_args The DeviceArray listing the arguments for the given kernel
		* @param training_dataset_size The number of samples used in the training process
		* @param ndims The number of attributes/features.
		**/
		void updateAlphas(cudaStream_t stream, GPUMLib::svm_kernel_type kernel_type,
			GPUMLib::DeviceMatrix<cudafloat> &d_x, GPUMLib::DeviceArray<cudafloat> &d_alphas,
			GPUMLib::DeviceArray<int> &d_y, cudafloat constant_c_negative, cudafloat constant_c_positive,
			GPUMLib::DeviceArray<cudafloat> &d_kernel_args, int training_dataset_size, int ndims) {
				switch (kernel_type) {
	case SVM_KT_LINEAR:
		cuUpdateAlphasAdvanced<SVM_KT_LINEAR> <<<1,1, 0, stream>>> (d_x.Pointer(),d_alphas.Pointer(),d_y.Pointer(),
			constant_c_negative, constant_c_positive, d_kernel_args.Pointer(),
			training_dataset_size, ndims);
		break;
	case SVM_KT_POLYNOMIAL:
		cuUpdateAlphasAdvanced<SVM_KT_POLYNOMIAL> <<<1,1, 0, stream>>> (d_x.Pointer(),d_alphas.Pointer(),d_y.Pointer(),
			constant_c_negative, constant_c_positive, d_kernel_args.Pointer(),
			training_dataset_size, ndims);
		break;
	case SVM_KT_RBF:
		cuUpdateAlphasAdvanced<SVM_KT_RBF> <<<1,1, 0, stream>>> (d_x.Pointer(),d_alphas.Pointer(),d_y.Pointer(),
			constant_c_negative, constant_c_positive, d_kernel_args.Pointer(),
			training_dataset_size, ndims);
		break;
	case SVM_KT_SIGMOID:
		cuUpdateAlphasAdvanced<SVM_KT_SIGMOID> <<<1,1, 0, stream>>> (d_x.Pointer(),d_alphas.Pointer(),d_y.Pointer(),
			constant_c_negative, constant_c_positive, d_kernel_args.Pointer(),
			training_dataset_size, ndims);
		break;
	case SVM_KT_UKF:
		cuUpdateAlphasAdvanced<SVM_KT_UKF> <<<1,1, 0, stream>>> (d_x.Pointer(),d_alphas.Pointer(),d_y.Pointer(),
			constant_c_negative, constant_c_positive, d_kernel_args.Pointer(),
			training_dataset_size, ndims);
		break;
				}
		}

		/**
		* Updates the KKT conditions.

		* @param stream A cudaStream_t context to associate this execution with
		* @param kernel_type The svm_kernel_type (enum) of used kernel (gaussian, linear, ukf, etc.)
		* @param n_blocks The number of CUDA blocks to execute
		* @param blocksize The number of threads in each block
		* @param d_f The array where to update the KKT conditions
		* @param d_y The DeviceArray containing the array of classes, one for each training sample
		* @param d_x The DeviceMatrix containing the attributes for each sample
		* @param d_kernel_args The DeviceArray listing the arguments for the given kernel
		* @param training_dataset_size The number of samples used in the training process
		* @param ndims The number of attributes/features.
		**/
		void updateKKTConditions(cudaStream_t stream, GPUMLib::svm_kernel_type kernel_type, int n_blocks, int blocksize,
			GPUMLib::DeviceArray<cudafloat> &d_f, GPUMLib::DeviceArray<int> &d_y,
			GPUMLib::DeviceMatrix<cudafloat> &d_x, GPUMLib::DeviceArray<cudafloat> &d_kernel_args,
			int training_dataset_size, int ndims) {
				switch (kernel_type) {
	case SVM_KT_LINEAR:
		cuUpdateKKTConditions<SVM_KT_LINEAR> <<< n_blocks, blocksize, 0, stream >>>(d_f.Pointer(), d_y.Pointer(), d_x.Pointer(),d_kernel_args.Pointer(),training_dataset_size,ndims);
		break;
	case SVM_KT_POLYNOMIAL:
		cuUpdateKKTConditions<SVM_KT_POLYNOMIAL><<< n_blocks, blocksize, 0, stream >>>(d_f.Pointer(), d_y.Pointer(), d_x.Pointer(),d_kernel_args.Pointer(),training_dataset_size,ndims);
		break;
	case SVM_KT_RBF:
		cuUpdateKKTConditions<SVM_KT_RBF><<< n_blocks, blocksize, 0, stream >>>(d_f.Pointer(), d_y.Pointer(), d_x.Pointer(),d_kernel_args.Pointer(),training_dataset_size,ndims);
		break;
	case SVM_KT_SIGMOID:
		cuUpdateKKTConditions<SVM_KT_SIGMOID><<< n_blocks, blocksize, 0, stream >>>(d_f.Pointer(), d_y.Pointer(), d_x.Pointer(),d_kernel_args.Pointer(),training_dataset_size,ndims);
		break;
	case SVM_KT_UKF:
		cuUpdateKKTConditions<SVM_KT_UKF><<< n_blocks, blocksize, 0, stream >>>(d_f.Pointer(), d_y.Pointer(), d_x.Pointer(),d_kernel_args.Pointer(),training_dataset_size,ndims);
		break;
				}
		}

		/**
		* Checks for CUDA errors occurred before this call. If an error occurred, it is printed to stdout.
		**/
		void checkCUDA_Errors() {
			// check for errors
			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				// print the CUDA error message and exit
				printf("CUDA error: %s\n", cudaGetErrorString(error));
				exit(-1);
			}
		}

		/** 
		* Launches the SMO training algorithm
		*
		* @param h_x A HostMatrix containing the training samples/patterns
		* @param h_y A HostArray containing the classes of the training samples
		* @param d_alphas A DeviceArray where to store the alpha/lagrange multiplier associated with each sample
		* @param constant_c_negative The penalization constant associated with the negative class (also used for the positive class)
		* @param constant_c_positive The penalization constant associated with the positive class (not used)
		* @param constant_epsilon The epsilon tolerance used in the first order search heuristic
		* @param constant_tau The threshold used for checking the duality gap/convergence
		* @param kernel_type The svm_kernel_type (enum) of used kernel (gaussian, linear, ukf, etc.)
		* @param kernel_args The DeviceArray listing the arguments for the given kernel
		* @param amount_threads The maximum number of threads used in each CUDA block
		**/
		void runSMO(GPUMLib::HostMatrix<cudafloat> & h_x, GPUMLib::HostArray<int> & h_y,
			GPUMLib::DeviceArray<cudafloat> & d_alphas,
			cudafloat constant_c_negative, cudafloat constant_c_positive,
			cudafloat constant_epsilon, cudafloat constant_tau, GPUMLib::svm_kernel_type kernel_type,
			cudafloat * kernel_args, int amount_threads) {

				int training_dataset_size = h_x.Rows();
				int ndims = h_x.Columns();

				if (DEBUG)
					cout << "started SMO..." << endl;

				int n_threads_per_block = NumberThreadsPerBlockThatBestFit(training_dataset_size, amount_threads);
				if (n_threads_per_block < 64)
					n_threads_per_block = 64;
				int n_blocks = NumberBlocks(training_dataset_size, n_threads_per_block);
				int n_threads_per_block_2ndpass = NumberThreadsPerBlockThatBestFit(n_blocks, n_threads_per_block);
				if (n_threads_per_block_2ndpass < 64)
					n_threads_per_block_2ndpass = 64;

				cout << "using " << n_blocks << " block(s) of " << n_threads_per_block << " threads each" << endl;
				cout << "using " << n_threads_per_block_2ndpass << " threads per block on second reduction" << endl;

				// create data structures in host's memory
				GPUMLib::HostArray < cudafloat > h_f(training_dataset_size);

				GPUMLib::HostArray < cudafloat > h_kernel_args(4);
				h_kernel_args[0] = (cudafloat) kernel_args[0];
				h_kernel_args[1] = (cudafloat) kernel_args[1];
				h_kernel_args[2] = (cudafloat) kernel_args[2];
				h_kernel_args[3] = (cudafloat) kernel_args[3];
				GPUMLib::DeviceArray < cudafloat > d_kernel_args(h_kernel_args);

				//copy to device
				GPUMLib::DeviceArray<int> d_y(h_y);
				GPUMLib::DeviceArray <cudafloat> d_f(h_f);
				GPUMLib::DeviceMatrix <cudafloat> d_x(h_x);

				cudafloat constant_c = min(constant_c_negative, constant_c_positive);

				//1. initialize
				int iteration = 0;
				//bhigh =  -1
				cudafloat h_b_high = CUDA_VALUE(-1.0);
				//blow = 1
				cudafloat h_b_low = CUDA_VALUE(1.0);

				//we can use the host for this, no need to use the GPU
				//ihigh = min{i : yi = 1}
				int i_high = findMinimumPositionTarget_HostArray(h_y, training_dataset_size, 1);

				//ilow = max{i : yi =  -1}
				int i_low = findMinimumPositionTarget_HostArray(h_y, training_dataset_size, -1);

				//make sure everything is OK
				if (i_high < 0 || i_low < 0) {
					cout << "Err: couldn't initialize SMO's indices.." << endl;
					assert(i_high >= 0);
					assert(i_low >= 0);
				}

				checkCUDA_Errors();
				GPUMLib::DeviceArray < cudafloat > d_minimums(n_blocks);
				GPUMLib::DeviceArray < cudafloat > d_maximuns(n_blocks);
				GPUMLib::DeviceArray<int> d_minimums_indices(n_blocks);
				GPUMLib::DeviceArray<int> d_maximums_indices(n_blocks);

				checkCUDA_Errors();
				cudaStream_t stream_memory_transaction;
				cudaStreamCreate(&stream_memory_transaction);
				cudaStream_t stream_kernel_execution = 0;
				//	cudaStreamCreate(&stream_kernel_execution);

				// initialize alphas and optimality conditions
				cuInitializeSMO<<< n_blocks, n_threads_per_block >>>(d_alphas.Pointer(), d_f.Pointer(), d_y.Pointer(), training_dataset_size);
				checkCUDA_Errors();

				//copy low/high indices to device's memory
				cudaMemcpyToSymbol(d_i_high, &i_high, sizeof(int));
				cudaMemcpyToSymbol(d_i_low, &i_low, sizeof(int));
				//copy low/high offsets to device's memory
				cudaMemcpyToSymbol(d_b_high, &h_b_high, sizeof(cudafloat));
				cudaMemcpyToSymbol(d_b_low, &h_b_low, sizeof(cudafloat));

				// update alphas before entering loops (only one thread required)
				updateAlphas(stream_kernel_execution, kernel_type, d_x, d_alphas, d_y, constant_c, constant_c, d_kernel_args, training_dataset_size, ndims);
				checkCUDA_Errors();

				//	cout << "----started SMO" << endl;
				bool converged = false;
				bool memory_transaction_requested = false;

				int last_i_low = -1;
				int last_i_high = -1;

				int convergence_checker = 0;

				// SMO loop
				for (;;) {
					if(convergence_checker>=8){
						convergence_checker=0;
						if (!memory_transaction_requested) {
							//copy low/high offsets from device's memory
							cudaMemcpyFromSymbolAsync(&h_b_high, d_b_high, sizeof(cudafloat), 0, cudaMemcpyDeviceToHost, stream_memory_transaction);
							cudaMemcpyFromSymbolAsync(&h_b_low, d_b_low, sizeof(cudafloat), 0, cudaMemcpyDeviceToHost, stream_memory_transaction);
							cudaMemcpyFromSymbolAsync(&i_low, d_i_low, sizeof(cudafloat), 0, cudaMemcpyDeviceToHost, stream_memory_transaction);
							cudaMemcpyFromSymbolAsync(&i_high, d_i_high, sizeof(cudafloat), 0, cudaMemcpyDeviceToHost, stream_memory_transaction);
							memory_transaction_requested = true;
							//			cudaStreamSynchronize(stream_memory_transaction);
						} else {
							if (cudaStreamQuery(stream_memory_transaction) == cudaSuccess) {
								memory_transaction_requested = false;

								// DEBUG
								//  			if ((iteration & 256) && !(iteration & 128) && !(iteration & 64) && !(iteration & 32) && !(iteration & 16) && !(iteration & 8) && !(iteration & 4)
								//  				&& !(iteration & 2) && !(iteration & 1)) {
								// 			cout << "iteration:" << iteration << "\tgap:" << h_b_low - h_b_high <<
								// 				"\tb_low:" << h_b_low << "\tb_high:" << h_b_high <<
								// 				"\ti_low:" << i_low << "\ti_high:" << i_high << endl;
								// 			}

								//two chosen alphas did not change for one iteration
								//TODO: this must be solved
								if(i_low==last_i_low && i_high==last_i_high){
									//unless the heuristic is changed, nothing to be done
									//terminate SMO
									converged = true;
								}
								last_i_low=i_low;
								last_i_high=i_high;

								if (h_b_low <= h_b_high + constant_tau || converged) {
									//					cout << "----SMO converged!" << endl;
									cout << "Converged! iteration:" << iteration << "\tgap:" << h_b_low - h_b_high << "\tb_low:" << h_b_low << "\tb_high:" << h_b_high << "\ti_low:"
										<< i_low << "\ti_high:" << i_high << endl;
									converged = true;
								}
							}
						}
					}
					convergence_checker++;
					//checkCUDA_Errors();

					if (converged)
						break;

					//check optimality conditions
					//update f_i for all i = 0...n-1
					updateKKTConditions(stream_kernel_execution, kernel_type, n_blocks, n_threads_per_block, d_f, d_y, d_x, d_kernel_args, training_dataset_size, ndims);
					checkCUDA_Errors();

					//compute b_high, i_high, b_low, i_low
					//using I_high and I_low sets
					kernelFirstOrderHeuristic1stPass(stream_kernel_execution, n_blocks, n_threads_per_block, d_f.Pointer(), d_alphas.Pointer(), d_y.Pointer(),
						d_minimums.Pointer(), d_minimums_indices.Pointer(), d_maximuns.Pointer(), d_maximums_indices.Pointer(), training_dataset_size, constant_epsilon,
						constant_c);
					checkCUDA_Errors();
					kernelFirstOrderHeuristicFinalPass(stream_kernel_execution, n_threads_per_block_2ndpass, d_minimums.Pointer(), d_minimums_indices.Pointer(),
						d_maximuns.Pointer(), d_maximums_indices.Pointer(), n_blocks);
					checkCUDA_Errors();

					//update the two lagrange multipliers
					updateAlphas(stream_kernel_execution, kernel_type, d_x, d_alphas, d_y, constant_c, constant_c, d_kernel_args, training_dataset_size, ndims);
					checkCUDA_Errors();

					iteration++;
				}

				puts("computing b");

				cudaStreamSynchronize(stream_memory_transaction);
				cudaStreamSynchronize(stream_kernel_execution);

				if (stream_memory_transaction != 0)
					cudaStreamDestroy(stream_memory_transaction);
				if (stream_kernel_execution != 0)
					cudaStreamDestroy(stream_kernel_execution);
				checkCUDA_Errors();

				cout << "total iterations: " << iteration << endl;
		}

		/** 
		* Launches the SVM training algorithm, calling internally the SMO and returning the array of Support Vectors.
		*
		* @param h_samples A HostMatrix containing the training samples/patterns
		* @param h_classes A HostArray containing the training samples' classes
		* @param constant_c_negative The penalization constant associated with the negative class (also used for the positive class)
		* @param constant_c_positive The penalization constant associated with the positive class (not used)
		* @param constant_epsilon The epsilon tolerance used in the first order search heuristic
		* @param constant_tau The threshold used for checking the duality gap/convergence
		* @param kernel_type The svm_kernel_type (enum) of used kernel (gaussian, linear, ukf, etc.)
		* @param kernel_args The DeviceArray listing the arguments for the given kernel
		* @param amount_threads The maximum number of threads used in each CUDA block

		* @param h_alphas The HostArray where to store the alphas for each sample after the training process
		* @param n_sv The The number of Support Vectors (samples with alphas > 0)
		* @param h_model The HostMatrix containing the SVM model (samples and alphas for each one)
		* @param h_b The hyperplane's bias (offset)
		**/
		void train(GPUMLib::HostMatrix<cudafloat> &h_samples, GPUMLib::HostArray<int> &h_classes,
			cudafloat constant_c_negative, cudafloat constant_c_positive,
			cudafloat constant_epsilon, cudafloat constant_tau,
			svm_kernel_type kernel_type, cudafloat * kernel_args,
			int amount_threads, GPUMLib::HostArray<cudafloat> &h_alphas,
			int &n_sv, GPUMLib::HostMatrix<cudafloat> &h_model,
			cudafloat &h_b ) 
		{
			//allocate array on the device to handle the alphas for each sample
			GPUMLib::DeviceArray<cudafloat> d_alphas(h_alphas);

			// "fire for effect!" (SMO)
			runSMO(h_samples, h_classes, d_alphas,
				constant_c_negative, constant_c_positive, constant_epsilon, constant_tau, 
				kernel_type, kernel_args, amount_threads);

			// copy alphas from device's memory to host
			h_alphas = d_alphas;

			//compress alphas array (only store alphas greater than 0)
			if (DEBUG)
				cout << "creating model..." << endl;
			int training_dataset_size = h_samples.Rows();
			int * alpha_indices = new int[training_dataset_size];
			n_sv = getSupportVectorIndices(h_alphas, alpha_indices, training_dataset_size);
			cout << n_sv << " SVs" << endl;

			// populate model matrix
			int ndims = h_samples.Columns();
			h_model.ResizeWithoutPreservingData(n_sv, ndims + 2);
			for (int row = 0; row < n_sv; row++) {
				//the index of current non zero alpha (support vector) on the original dataset
				int index = alpha_indices[row];
				//the value of alpha (lagrange multiplier)
				cudafloat alpha_i = h_alphas[index];
				//set alpha on model
				h_model(row, 0) = alpha_i;
				//the class associated with current alpha
				int c_i = h_classes[index];
				//set class on model
				h_model(row, 1) = (cudafloat) c_i;
				//set the remaining elements as the features
				for (int feature_i = 0; feature_i < ndims; feature_i++) {
					//get the original attribute
					cudafloat attribute = h_samples(index, feature_i);
					//copy to the model
					h_model(row, feature_i + 2) = attribute;
				}
			}
			delete alpha_indices;

			cudaStream_t stream_bias_calculus;
			cudaStreamCreate(&stream_bias_calculus);

			GPUMLib::DeviceMatrix<cudafloat> d_model(h_model);
			// compute bias using model
			GPUMLib::HostArray<cudafloat> h_offsets(n_sv);
			GPUMLib::DeviceArray<cudafloat> d_offsets(h_offsets);

			GPUMLib::HostArray<cudafloat> h_kernel_args(4);
			h_kernel_args[0] = (cudafloat) kernel_args[0];
			h_kernel_args[1] = (cudafloat) kernel_args[1];
			h_kernel_args[2] = (cudafloat) kernel_args[2];
			h_kernel_args[3] = (cudafloat) kernel_args[3];
			GPUMLib::DeviceArray<cudafloat> d_kernel_args(h_kernel_args);

			int n_threads_per_block = NumberThreadsPerBlockThatBestFit(n_sv, amount_threads);
			int n_blocks = NumberBlocks(n_sv, n_threads_per_block);
			int n_threads_per_block_2ndpass = NumberThreadsPerBlockThatBestFit(n_blocks, n_threads_per_block);
			if(n_threads_per_block_2ndpass < 64)
				n_threads_per_block_2ndpass = 64;

			//printf("n_blocks:%d\tn_threads_per_block:%d\tn_threads_per_block_2ndpass:%d\n",n_blocks,n_threads_per_block,n_threads_per_block_2ndpass);

			//compute offset on the GPU
			switch (kernel_type){
	case SVM_KT_LINEAR:
		cuCalculateOffsetsUsingModel<SVM_KT_LINEAR> <<<n_blocks, n_threads_per_block, 0, stream_bias_calculus>>>
			(d_offsets.Pointer(), d_model.Pointer(), n_sv, ndims, d_kernel_args.Pointer());
		break;
	case SVM_KT_POLYNOMIAL:
		cuCalculateOffsetsUsingModel<SVM_KT_POLYNOMIAL> <<<n_blocks, n_threads_per_block, 0, stream_bias_calculus>>>
			(d_offsets.Pointer(), d_model.Pointer(), n_sv, ndims, d_kernel_args.Pointer());
		break;
	case SVM_KT_RBF:
		cuCalculateOffsetsUsingModel<SVM_KT_RBF> <<<n_blocks, n_threads_per_block, 0, stream_bias_calculus>>>
			(d_offsets.Pointer(), d_model.Pointer(), n_sv, ndims, d_kernel_args.Pointer());
		break;
	case SVM_KT_SIGMOID:
		cuCalculateOffsetsUsingModel<SVM_KT_SIGMOID> <<<n_blocks, n_threads_per_block, 0, stream_bias_calculus>>>
			(d_offsets.Pointer(), d_model.Pointer(), n_sv, ndims, d_kernel_args.Pointer());
		break;
	case SVM_KT_UKF:
		cuCalculateOffsetsUsingModel<SVM_KT_UKF> <<<n_blocks, n_threads_per_block, 0, stream_bias_calculus>>>
			(d_offsets.Pointer(), d_model.Pointer(), n_sv, ndims, d_kernel_args.Pointer());
		break;
			}
			checkCUDA_Errors();

			h_offsets = d_offsets;

			GPUMLib::DeviceArray<cudafloat> d_partialsums(n_blocks);

			calculateB_1stPass(stream_bias_calculus, n_blocks, n_threads_per_block, d_offsets.Pointer(), d_partialsums.Pointer(), n_sv);
			cudaStreamSynchronize(stream_bias_calculus);
			calculateB_FinalPass(stream_bias_calculus, n_threads_per_block_2ndpass, d_partialsums.Pointer(), n_blocks);
			cudaStreamSynchronize(stream_bias_calculus);
			cudaMemcpyFromSymbol(&h_b, d_b, sizeof(cudafloat), 0, cudaMemcpyDeviceToHost);
			h_b = h_b / n_sv;
			cout << "bias: " << h_b << " nSVs: " << n_sv << endl;

			if(stream_bias_calculus != 0){
				cudaStreamSynchronize(stream_bias_calculus);
				cudaStreamDestroy(stream_bias_calculus);
			}

			//TODO: that shared memory bug
			//		checkCUDA_Errors();
		}

		/** 
		* Launches the SVM classification algorithm.
		*
		* @param h_model The HostMatrix containing the SVM model (samples and alphas for each one)
		* @param h_testing_samples A HostMatrix containing the samples/patterns to be classified
		* @param kernel_args The DeviceArray listing the arguments for the given kernel
		* @param amount_threads The maximum number of threads used in each CUDA block
		* @param kernel_type The svm_kernel_type (enum) of used kernel (gaussian, linear, ukf, etc.)
		* @param n_sv The The number of Support Vectors (samples with alphas > 0) in the model
		* @param h_b The hyperplane's bias (offset)
		* @param ndims The number of dimensions in the training set (size of each sample)
		* @param h_testing_results The HostArray where to store the classification results.
		**/
		void classify(GPUMLib::HostMatrix<cudafloat> &h_model, GPUMLib::HostMatrix<cudafloat> &h_testing_samples,
			cudafloat * kernel_args, int amount_threads,
			GPUMLib::svm_kernel_type kernel_type, int n_sv, cudafloat h_b, int ndims,
			GPUMLib::HostArray<int> &h_testing_results ) 
		{
			//create GPU structures
			GPUMLib::DeviceMatrix<cudafloat> d_model(h_model);
			GPUMLib::DeviceMatrix<cudafloat> d_testing_samples(h_testing_samples);
			GPUMLib::DeviceArray<int> d_testing_results(h_testing_results);

			GPUMLib::HostArray<cudafloat> h_kernel_args(4);
			h_kernel_args[0] = (cudafloat) kernel_args[0];
			h_kernel_args[1] = (cudafloat) kernel_args[1];
			h_kernel_args[2] = (cudafloat) kernel_args[2];
			h_kernel_args[3] = (cudafloat) kernel_args[3];
			GPUMLib::DeviceArray<cudafloat> d_kernel_args(h_kernel_args);

			int testing_dataset_size=h_testing_samples.Rows();
			int n_threads_per_block = NumberThreadsPerBlockThatBestFit(testing_dataset_size, amount_threads);
			int n_blocks = NumberBlocks(testing_dataset_size, n_threads_per_block);

			//process dataset on the GPU
			switch (kernel_type){
	case SVM_KT_LINEAR:
		cuClassifyDataSet<SVM_KT_LINEAR> <<<n_blocks, n_threads_per_block>>>
			(d_testing_results.Pointer(), d_testing_samples.Pointer(), testing_dataset_size, d_model.Pointer(), n_sv, h_b, ndims, d_kernel_args.Pointer());
		break;
	case SVM_KT_POLYNOMIAL:
		cuClassifyDataSet<SVM_KT_POLYNOMIAL> <<<n_blocks, n_threads_per_block>>>(d_testing_results.Pointer(), d_testing_samples.Pointer(), testing_dataset_size, d_model.Pointer(), n_sv, h_b, ndims, d_kernel_args.Pointer());
		break;
	case SVM_KT_RBF:
		cuClassifyDataSet<SVM_KT_RBF> <<<n_blocks, n_threads_per_block>>>(d_testing_results.Pointer(), d_testing_samples.Pointer(), testing_dataset_size, d_model.Pointer(), n_sv, h_b, ndims, d_kernel_args.Pointer());
		break;
	case SVM_KT_SIGMOID:
		cuClassifyDataSet<SVM_KT_SIGMOID> <<<n_blocks, n_threads_per_block>>>(d_testing_results.Pointer(), d_testing_samples.Pointer(), testing_dataset_size, d_model.Pointer(), n_sv, h_b, ndims, d_kernel_args.Pointer());
		break;
	case SVM_KT_UKF:
		cuClassifyDataSet<SVM_KT_UKF> <<<n_blocks, n_threads_per_block>>>(d_testing_results.Pointer(), d_testing_samples.Pointer(), testing_dataset_size, d_model.Pointer(), n_sv, h_b, ndims, d_kernel_args.Pointer());
		break;
			}
			//copy classification results from device to host
			h_testing_results = d_testing_results;
		}

	};

	//! \example svm_example.cu
	//! Example of the SVM algorithm usage.
	
	//! @}

} //namespace

#endif
