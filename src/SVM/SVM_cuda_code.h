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

#ifndef SVM_CUDA_H_
#define SVM_CUDA_H_

//GPUMLib stuff
#include <cuda.h>

namespace GPUMLib {

	//class SVM {
	// allocate SMO variables on the device

	//! GPU's index of SMO current alpha associated with the low working index
	__device__ int d_i_low;
	//! GPU's index of SMO current alpha associated with the high working index
	__device__ int d_i_high;

	//! GPU's optimality condition for the sample associated with the low working index
	__device__ cudafloat d_b_low;
	__device__ cudafloat d_b_high;

	//! GPU's SMO old low alpha
	__device__ cudafloat d_alpha_i_low_old;
	//! GPU's SMO old high alpha
	__device__ cudafloat d_alpha_i_high_old;

	//! GPU's SMO new low alpha
	__device__ cudafloat d_alpha_i_low_new;
	//! GPU's SMO new high alpha
	__device__ cudafloat d_alpha_i_high_new;

	//! GPU's hyperplane offset
	__device__ cudafloat d_b;

	//! Gets the calling thread's ID in CUDA
	//! \param Returns the calling thread's ID
	__device__ int cuGetThreadID() {
		return blockIdx.x * blockDim.x + threadIdx.x;
	}

	//! Gets the value of a attribute/feature
	//! \param samples The matrix of samples
	//! \param n_samples The number of samples contained in the samples matrix
	//! \param attribute_id The number of the attribute to get the value
	//! \return The attribute's value
	__device__ cudafloat cuGetSampleAttribute(cudafloat * samples,
		int n_samples, int sample_id,
		int attribute_id) {
			int index = n_samples * attribute_id + sample_id;
			return samples[index];
	}

	template <svm_kernel_type kernel_type>
	__device__ cudafloat cuKernelDotProductUsingModelOnly(int sv_i, int sample_i,
		cudafloat * model, int nsvs,
		int num_dimensions,
		cudafloat * kernel_args) {
			// select kernel_type from available_kernels
			if(kernel_type==SVM_KT_LINEAR) {
				// 0 = linear kernel (default)
				// = x1.x2
				cudafloat sum = CUDA_VALUE(0.0);
				for (int i = 0; i < num_dimensions; i++) {
					cudafloat x0_i = cuGetSampleAttribute(model, nsvs, sv_i, i + 2);
					cudafloat x1_i = cuGetSampleAttribute(model, nsvs, sample_i, i + 2);
					sum += x0_i * x1_i;
				}
				return sum;
			} else if(kernel_type==SVM_KT_POLYNOMIAL) {
				//polynomial kernel
				//(a*(x1.x2)+b)^c
				// sum = x1.x2
				cudafloat sum = CUDA_VALUE(0.0);
				for (int i = 0; i < num_dimensions; i++) {
					cudafloat x0_i = cuGetSampleAttribute(model, nsvs, sv_i, i + 2);
					cudafloat x1_i = cuGetSampleAttribute(model, nsvs, sample_i, i + 2);
					sum += x0_i * x1_i;
				}
				cudafloat val = CUDA_POW(kernel_args[0] * sum + kernel_args[1], kernel_args[2]);
				return val;
			} else if(kernel_type==SVM_KT_RBF) {
				// radial basis function (RBF) kernel, a = sigma
				// e^(-(1/(a^2)*(x1-x2)^2))
				cudafloat sum_dif_squared = CUDA_VALUE(0.0);
				for (int i = 0; i < num_dimensions; i++) {
					cudafloat x0_i = cuGetSampleAttribute(model, nsvs, sv_i, i + 2);
					cudafloat x1_i = cuGetSampleAttribute(model, nsvs, sample_i, i + 2);
					cudafloat _dif = x0_i - x1_i;
					cudafloat _dif_sq = _dif * _dif;
					sum_dif_squared += _dif_sq;
				}
				cudafloat result = CUDA_EXP(-kernel_args[0] * sum_dif_squared);
				return result;
			} else if(kernel_type==SVM_KT_SIGMOID) {
				// sigmoid kernel
				cudafloat sum = CUDA_VALUE(0.0);
				for (int i = 0; i < num_dimensions; i++) {
					cudafloat x0_i = cuGetSampleAttribute(model, nsvs, sv_i, i + 2);
					cudafloat x1_i = cuGetSampleAttribute(model, nsvs, sample_i, i + 2);
					sum += x0_i * x1_i;
				}
				cudafloat val = CUDA_TANH(kernel_args[0] * sum + kernel_args[1]);
				return val;
			} else if(kernel_type==SVM_KT_UKF) {
				// universal kernel function
				// K(x1,x2) = a*(||x1-x2||^2+b^2)^-c
				cudafloat sum_dif_squared = 0;
				for (int i = 0; i < num_dimensions; i++) {
					cudafloat x0_i = cuGetSampleAttribute(model, nsvs, sv_i, i + 2);
					cudafloat x1_i = cuGetSampleAttribute(model, nsvs, sample_i, i + 2);
					cudafloat _dif = x0_i - x1_i;
					cudafloat _dif_sq = _dif * _dif;
					sum_dif_squared += _dif_sq;
				}
				cudafloat result = kernel_args[0] * CUDA_POW(sum_dif_squared + kernel_args[1] * kernel_args[1], -kernel_args[2]);
				return result;
			} else
				return CUDA_VALUE(0.0);
	}

	template <svm_kernel_type kernel_type>
	__device__ cudafloat cuKernelDotProduct(int sv_i, int sample_i,
		cudafloat * model, cudafloat * dataset,
		int nsvs, int dataset_size,
		int num_dimensions,
		cudafloat * kernel_args) {
			// select kernel_type from available_kernels
			if(kernel_type==SVM_KT_LINEAR) {
				// 0 = linear kernel (default)
				// = x1.x2
				cudafloat sum = CUDA_VALUE(0.0);
				for (int i = 0; i < num_dimensions; i++) {
					cudafloat x0_i = cuGetSampleAttribute(model, nsvs, sv_i, i + 2);
					cudafloat x1_i = cuGetSampleAttribute(dataset, dataset_size, sample_i, i);
					sum += x0_i * x1_i;
				}
				return sum;
			} else if(kernel_type==SVM_KT_POLYNOMIAL) {
				//polynomial kernel
				//(a*(x1.x2)+b)^c
				// sum = x1.x2
				cudafloat sum = CUDA_VALUE(0.0);
				for (int i = 0; i < num_dimensions; i++) {
					cudafloat x0_i = cuGetSampleAttribute(model, nsvs, sv_i, i + 2);
					cudafloat x1_i = cuGetSampleAttribute(dataset, dataset_size, sample_i, i);
					sum += x0_i * x1_i;
				}
				cudafloat val = CUDA_POW(kernel_args[0] * sum + kernel_args[1], kernel_args[2]);
				return val;
			} else if(kernel_type==SVM_KT_RBF) {
				// radial basis function (RBF) kernel, a = sigma
				// e^(-(1/(a^2)*(x1-x2)^2))
				cudafloat sum_dif_squared = CUDA_VALUE(0.0);
				for (int i = 0; i < num_dimensions; i++) {
					cudafloat x0_i = cuGetSampleAttribute(model, nsvs, sv_i, i + 2);
					cudafloat x1_i = cuGetSampleAttribute(dataset, dataset_size, sample_i, i);
					cudafloat _dif = x0_i - x1_i;
					cudafloat _dif_sq = _dif * _dif;
					sum_dif_squared += _dif_sq;
				}
				cudafloat result = CUDA_EXP(-kernel_args[0] * sum_dif_squared);
				return result;
			} else if(kernel_type==SVM_KT_SIGMOID) {
				// sigmoid kernel
				cudafloat sum = CUDA_VALUE(0.0);
				for (int i = 0; i < num_dimensions; i++) {
					cudafloat x0_i = cuGetSampleAttribute(model, nsvs, sv_i, i + 2);
					cudafloat x1_i = cuGetSampleAttribute(dataset, dataset_size, sample_i, i);
					sum += x0_i * x1_i;
				}
				cudafloat val = CUDA_TANH(kernel_args[0] * sum + kernel_args[1]);
				return val;
			} else if(kernel_type==SVM_KT_UKF) {
				// universal kernel function
				// K(x1,x2) = a*(||x1-x2||^2+b^2)^-c
				cudafloat sum_dif_squared = 0;
				for (int i = 0; i < num_dimensions; i++) {
					cudafloat x0_i = cuGetSampleAttribute(model, nsvs, sv_i, i + 2);
					cudafloat x1_i = cuGetSampleAttribute(dataset, dataset_size, sample_i, i);
					cudafloat _dif = x0_i - x1_i;
					cudafloat _dif_sq = _dif * _dif;
					sum_dif_squared += _dif_sq;
				}
				cudafloat result = kernel_args[0] * CUDA_POW(sum_dif_squared + kernel_args[1] * kernel_args[1], -kernel_args[2]);
				return result;
			} else
				return CUDA_VALUE(0.0);
	}

	template <svm_kernel_type kernel_type>
	__device__ cudafloat cuKernelDotProduct(int i0, int i1, cudafloat * samples, int n_samples, int num_dimensions, cudafloat * kernel_args) {
		// select kernel_type from available_kernels
		if(kernel_type==SVM_KT_LINEAR) {
			// 0 = linear kernel (default)
			// = x1.x2
			cudafloat sum = CUDA_VALUE(0.0);
			for (int i = 0; i < num_dimensions; i++) {
				cudafloat x0_i = cuGetSampleAttribute(samples, n_samples, i0, i);
				cudafloat x1_i = cuGetSampleAttribute(samples, n_samples, i1, i);
				sum += x0_i * x1_i;
			}
			return sum;
		} else if(kernel_type==SVM_KT_POLYNOMIAL) {
			//polynomial kernel
			//(a*(x1.x2)+b)^c
			// sum = x1.x2
			cudafloat sum = CUDA_VALUE(0.0);
			for (int i = 0; i < num_dimensions; i++) {
				cudafloat x0_i = cuGetSampleAttribute(samples, n_samples, i0, i);
				cudafloat x1_i = cuGetSampleAttribute(samples, n_samples, i1, i);
				sum += x0_i * x1_i;
			}
			cudafloat val = CUDA_POW(kernel_args[0] * sum + kernel_args[1], kernel_args[2]);
			return val;
		} else if(kernel_type==SVM_KT_RBF) {
			// radial basis function (RBF) kernel, a = sigma
			// e^(-(1/(a^2)*(x1-x2)^2))
			cudafloat sum_dif_squared = CUDA_VALUE(0.0);
			for (int i = 0; i < num_dimensions; i++) {
				cudafloat x0_i = cuGetSampleAttribute(samples, n_samples, i0, i);
				cudafloat x1_i = cuGetSampleAttribute(samples, n_samples, i1, i);
				cudafloat _dif = x0_i - x1_i;
				cudafloat _dif_sq = _dif * _dif;
				sum_dif_squared += _dif_sq;
			}
			cudafloat result = CUDA_EXP(-kernel_args[0] * sum_dif_squared);
			return result;
		} else if(kernel_type==SVM_KT_SIGMOID) {
			// sigmoid kernel
			cudafloat sum = CUDA_VALUE(0.0);
			for (int i = 0; i < num_dimensions; i++) {
				cudafloat x0_i = cuGetSampleAttribute(samples, n_samples, i0, i);
				cudafloat x1_i = cuGetSampleAttribute(samples, n_samples, i1, i);
				sum += x0_i * x1_i;
			}
			cudafloat val = CUDA_TANH(kernel_args[0] * sum + kernel_args[1]);
			return val;
		} else if(kernel_type==SVM_KT_UKF) {
			// universal kernel function
			// K(x1,x2) = a*(||x1-x2||^2+b^2)^-c
			cudafloat sum_dif_squared = 0;
			for (int i = 0; i < num_dimensions; i++) {
				cudafloat x0_i = cuGetSampleAttribute(samples, n_samples, i0, i);
				cudafloat x1_i = cuGetSampleAttribute(samples, n_samples, i1, i);
				cudafloat _dif = x0_i - x1_i;
				cudafloat _dif_sq = _dif * _dif;
				sum_dif_squared += _dif_sq;
			}
			cudafloat result = kernel_args[0] * CUDA_POW(sum_dif_squared + kernel_args[1] * kernel_args[1], -kernel_args[2]);
			return result;
		} else
			return CUDA_VALUE(0.0);
	}

	template <svm_kernel_type kernel_type>
	__global__ void cuUpdateKKTConditions(cudafloat * f, int * y, cudafloat * x, cudafloat * kernel_args, int nsamples, int ndims) {
		int tid = cuGetThreadID();
		if(tid<nsamples) {
			cudafloat kxh_xi = cuKernelDotProduct<kernel_type>(d_i_high,tid,x,nsamples,ndims,kernel_args);
			cudafloat kxl_xi = cuKernelDotProduct<kernel_type>(d_i_low,tid,x,nsamples,ndims,kernel_args);
			//do the update
			cudafloat fdelta = (d_alpha_i_high_new - d_alpha_i_high_old) * y[d_i_high] * kxh_xi + (d_alpha_i_low_new - d_alpha_i_low_old) * y[d_i_low] * kxl_xi;
			f[tid] += fdelta;
		}
	}

	template <svm_kernel_type kernel_type>
	__device__ int cuClassify(int sample_index, cudafloat * dataset, int dataset_size, cudafloat * model, int nsvs, cudafloat b, int ndims, cudafloat * kernel_args ) {
		cudafloat sum = CUDA_VALUE(0.0);
		for (int sv_i = 0; sv_i < nsvs; sv_i++) {
			cudafloat alpha_i = cuGetSampleAttribute(model, nsvs, sv_i, 0);
			cudafloat y = cuGetSampleAttribute(model, nsvs, sv_i, 1);
			cudafloat k_proj = cuKernelDotProduct<kernel_type>(sv_i, sample_index, model, dataset, nsvs, dataset_size, ndims, kernel_args);
			sum += alpha_i * y * k_proj;
		}
		sum += b;
		if(sum > 0)
			return 1;
		else
			return -1;
	}

	template <svm_kernel_type kernel_type>
	__global__ void cuClassifyDataSet(int * results, cudafloat * dataset, int dataset_size, cudafloat * model, int nsvs, cudafloat b, int ndims, cudafloat * kernel_args) {
		int tid = cuGetThreadID();
		if(tid < dataset_size) {
			int result = cuClassify<kernel_type>(tid, dataset, dataset_size, model, nsvs, b, ndims, kernel_args);
			results[tid] = result;
		}
	}

	template <svm_kernel_type kernel_type>
	__device__ cudafloat cuClassifierOutput(int sample_index, cudafloat * model, int nsvs, int ndims, cudafloat * kernel_args ) {
		cudafloat sum = CUDA_VALUE(0.0);
		for (int sv_i = 0; sv_i < nsvs; sv_i++) {
			cudafloat alpha_i = cuGetSampleAttribute(model, nsvs, sv_i, 0);
			cudafloat y = cuGetSampleAttribute(model, nsvs, sv_i, 1);
			cudafloat k_proj = cuKernelDotProductUsingModelOnly<kernel_type>(sv_i, sample_index, model, nsvs, ndims, kernel_args);
			sum += alpha_i * y * k_proj;
		}
		return sum;
	}

	template <svm_kernel_type kernel_type>
	__global__ void cuCalculateOffsetsUsingModel(cudafloat * results, cudafloat * model, int nsvs, int ndims, cudafloat * kernel_args) {
		int tid = cuGetThreadID();
		if(tid < nsvs) {
			cudafloat result = cuClassifierOutput<kernel_type>(tid, model, nsvs, ndims, kernel_args);
			results[tid] = result;
		}
	}

	template <int blockSize>
	__global__ void cuCalculateB_1stPass(cudafloat * offsets, cudafloat * bs, int n_svs) {
		// local arrays (shared memory)
		extern __shared__ cudafloat results[];

		// compute classification
		int tid = cuGetThreadID();
		cudafloat result = CUDA_VALUE(0.0);
		if(tid < n_svs) {
			result = offsets[tid];
		}
		results[threadIdx.x] = result;
		__syncthreads();

		if (blockSize >= 1024) {
			if (threadIdx.x < 512) {
				results[threadIdx.x] += results[threadIdx.x + 512];
			}
			__syncthreads();
		}

		if (blockSize >= 512) {
			if (threadIdx.x < 256) {
				results[threadIdx.x] += results[threadIdx.x + 256];
			}
			__syncthreads();
		}

		if (blockSize >= 256) {
			if (threadIdx.x < 128) {
				results[threadIdx.x] += results[threadIdx.x + 128];
			}
			__syncthreads();
		}

		if (blockSize >= 128) {
			if (threadIdx.x < 64) {
				results[threadIdx.x] += results[threadIdx.x + 64];
			}
			__syncthreads();
		}

		if (threadIdx.x < 32) {
			volatile cudafloat * _results = results;

			if (blockSize >= 64) {
				_results[threadIdx.x] += _results[threadIdx.x + 32];
			}
			if (blockSize >= 32) {
				_results[threadIdx.x] += _results[threadIdx.x + 16];
			}
			if (blockSize >= 16) {
				_results[threadIdx.x] += _results[threadIdx.x + 8];
			}
			if (blockSize >= 8) {
				_results[threadIdx.x] += _results[threadIdx.x + 4];
			}
			if (blockSize >= 4) {
				_results[threadIdx.x] += _results[threadIdx.x + 2];
			}
			if (blockSize >= 2) {
				_results[threadIdx.x] += _results[threadIdx.x + 1];
			}

			if (threadIdx.x == 0) {
				bs[blockIdx.x] = results[0];
			}
		}
	}

	template<int blockSize>
	KERNEL cuCalculateB_FinalPass(cudafloat * data, int size) {
		extern __shared__ cudafloat
			sum[];

		cudafloat value = CUDA_VALUE(0.0);
		for (int i = threadIdx.x; i < size; i += blockDim.x)
			value += data[i];
		sum[threadIdx.x] = value;
		__syncthreads();

		if (blockSize >= 1024) {
			if (threadIdx.x < 512)
				sum[threadIdx.x] += sum[threadIdx.x + 512];
			__syncthreads();
		}

		if (blockSize >= 512) {
			if (threadIdx.x < 256)
				sum[threadIdx.x] += sum[threadIdx.x + 256];
			__syncthreads();
		}

		if (blockSize >= 256) {
			if (threadIdx.x < 128)
				sum[threadIdx.x] += sum[threadIdx.x + 128];
			__syncthreads();
		}

		if (blockSize >= 128) {
			if (threadIdx.x < 64)
				sum[threadIdx.x] += sum[threadIdx.x + 64];
			__syncthreads();
		}

		if (threadIdx.x < 32) {
			volatile cudafloat * _sum = sum;

			if (blockSize >= 64)
				_sum[threadIdx.x] += _sum[threadIdx.x + 32];
			if (blockSize >= 32)
				_sum[threadIdx.x] += _sum[threadIdx.x + 16];
			if (blockSize >= 16)
				_sum[threadIdx.x] += _sum[threadIdx.x + 8];
			if (blockSize >= 8)
				_sum[threadIdx.x] += _sum[threadIdx.x + 4];
			if (blockSize >= 4)
				_sum[threadIdx.x] += _sum[threadIdx.x + 2];
			if (blockSize >= 2)
				_sum[threadIdx.x] += _sum[threadIdx.x + 1];

			if (threadIdx.x == 0) {
				d_b = sum[0];
			}
		}
	}

	/************************************************************************/
	/* for all i=0...n-1
	alpha_i=0 (and old alphas)
	f_i=-y_i
	*/
	/************************************************************************/
	__global__ void cuInitializeSMO(cudafloat * alphas, cudafloat * f, int * classes, int _nsamples) {
		int tid = cuGetThreadID();
		// for all i = 0... (N-1)
		if (tid < _nsamples) {
			//alpha_i=0
			alphas[tid] = CUDA_VALUE(0.0);
			//f_i=-y_i
			f[tid] = -classes[tid];
		}
	}

	template <svm_kernel_type kernel_type>
	__global__ void cuUpdateAlphasSimple(cudafloat * x, cudafloat * alphas, int * y,
		cudafloat constant_c_negative, cudafloat constant_c_positive, cudafloat * kernel_args,
		int nsamples, int ndims) {
			// get thread ID
			int tid = cuGetThreadID();
			// only one thread executes this
			if(tid == 0) {

				// new alphas
				cudafloat ailn;
				cudafloat aihn;

				//copy data from device's memory to local registers
				int il = d_i_low;
				int ih = d_i_high;
				cudafloat aiho = alphas[ih];
				cudafloat ailo = alphas[il];
				cudafloat bl = d_b_low;
				cudafloat bh = d_b_high;

				// targets
				int y_i_low = y[il];
				int y_i_high = y[ih];

				// store old alphas for this iteration
				d_alpha_i_low_old = ailo;
				d_alpha_i_high_old = aiho;

				// kernel computations
				cudafloat kxl_xl = cuKernelDotProduct<kernel_type>(il,il,x,nsamples,ndims,kernel_args);
				cudafloat kxh_xh = cuKernelDotProduct<kernel_type>(ih,ih,x,nsamples,ndims,kernel_args);
				cudafloat kxh_xl = cuKernelDotProduct<kernel_type>(ih,il,x,nsamples,ndims,kernel_args);

				// eta
				cudafloat eta = kxh_xh + kxl_xl - CUDA_VALUE(2.0) * kxh_xl;

				// compute new alphas
				ailn = d_alpha_i_low_old + y_i_low * (bh - bl) / eta;
				aihn = d_alpha_i_high_old + y_i_low * y_i_high * (d_alpha_i_low_old - ailn);

				// clip alphas in range 0 <= a <= C
				if (aihn < 0.0) {
					aihn = CUDA_VALUE(0.0);
				} else if (y_i_high == -1 && aihn > constant_c_negative) {
					aihn = constant_c_negative;
				} else if (y_i_high == 1 && aihn > constant_c_positive) {
					aihn = constant_c_positive;
				}

				if (ailn < CUDA_VALUE(0.0)) {
					ailn = CUDA_VALUE(0.0);
				} else if (y_i_low == -1 && ailn > constant_c_negative) {
					ailn = constant_c_negative;
				} else if (y_i_low == 1 && ailn > constant_c_positive) {
					ailn = constant_c_positive;
				}

				//store new alphas from registers back into device's memory
				alphas[il] = ailn;
				d_alpha_i_low_new = ailn;
				alphas[ih] = aihn;
				d_alpha_i_high_new = aihn;
			}
	}

	// updates alphas. Should be only executed by one thread (grid with one thread)
	template <svm_kernel_type kernel_type>
	__global__ void cuUpdateAlphasAdvanced(cudafloat * x, cudafloat * alphas, int * y,
		cudafloat constant_c_negative, cudafloat constant_c_positive, cudafloat * kernel_args,
		int nsamples, int ndims) {
			// get thread ID
			int tid = cuGetThreadID();
			// only one thread executes this
			if(tid == 0) {

				//ARGH
				cudafloat constant_c = constant_c_negative;

				// new alphas
				cudafloat ailn;
				cudafloat aihn;

				//copy data from device's memory to local registers
				int il = d_i_low;
				int ih = d_i_high;
				cudafloat aiho = alphas[ih];
				cudafloat ailo = alphas[il];
				cudafloat bl = d_b_low;
				cudafloat bh = d_b_high;

				// targets
				cudafloat y_i_low = y[il];
				cudafloat y_i_high = y[ih];

				// store old alphas for this iteration
				d_alpha_i_low_old = ailo;
				d_alpha_i_high_old = aiho;

				// kernel computations
				cudafloat kxl_xl = cuKernelDotProduct<kernel_type>(il,il,x,nsamples,ndims,kernel_args);
				cudafloat kxh_xh = cuKernelDotProduct<kernel_type>(ih,ih,x,nsamples,ndims,kernel_args);
				cudafloat kxh_xl = cuKernelDotProduct<kernel_type>(ih,il,x,nsamples,ndims,kernel_args);

				// eta
				cudafloat eta = kxh_xh + kxl_xl - CUDA_VALUE(2.0) * kxh_xl;

				cudafloat alphadiff = ailo - aiho;
				cudafloat sign = y_i_low * y_i_high;

				cudafloat alpha_l_upperbound, alpha_l_lowerbound;
				if (sign < CUDA_VALUE(0.0)) {
					if (alphadiff < 0) {
						alpha_l_lowerbound = 0;
						alpha_l_upperbound = constant_c + alphadiff;
					} else {
						alpha_l_lowerbound = alphadiff;
						alpha_l_upperbound = constant_c;
					}
				} else {
					double alpha_sum = ailo + aiho;
					if (alpha_sum < constant_c) {
						alpha_l_upperbound = alpha_sum;
						alpha_l_lowerbound = 0;
					} else {
						alpha_l_lowerbound = alpha_sum - constant_c;
						alpha_l_upperbound = constant_c;
					}
				}
				if (eta > 0) {
					ailn = ailo + y_i_low * (bh - bl) / eta;
					if (ailn < alpha_l_lowerbound) {
						ailn = alpha_l_lowerbound;
					} else
						if (ailn > alpha_l_upperbound) {
							ailn = alpha_l_upperbound;
						}
				} else {
					double slope = y_i_low * (bh - bl);
					double delta = slope * (alpha_l_upperbound - alpha_l_lowerbound);
					if (delta > 0) {
						if (slope > 0) {
							ailn = alpha_l_upperbound;
						} else {
							ailn = alpha_l_lowerbound;
						}
					} else {
						ailn = ailo;
					}
				}
				cudafloat alpha_l_diff = ailn - ailo;
				cudafloat alpha_h_diff = -sign * alpha_l_diff;
				aihn = aiho + alpha_h_diff;

				//store new alphas from registers back into device's memory
				alphas[il] = ailn;
				d_alpha_i_low_new = ailn;
				alphas[ih] = aihn;
				d_alpha_i_high_new = aihn;
			}
	}

	/**
	* checks if the difference between two floats is within a tolerance
	*/
	inline __device__ bool cufequal(double a, double b, double tolerance) {
		double dif = fabs(a - b);
		return dif < tolerance;
	}

	template<int blockSize> KERNEL cuFirstOrderHeuristic1stPass(cudafloat * f, cudafloat * alphas, int * y, cudafloat * minimuns, int * min_indices,
		cudafloat * maximuns, int * max_indices, int input_size, cudafloat constant_epsilon, cudafloat constant_c) {
			// local arrays (shared memory)
			extern __shared__ cudafloat	minvalues[];
			int * minposs = (int *) (minvalues + blockSize);
			//-------------
			cudafloat * maxvalues = (cudafloat*) (minposs + blockSize);
			int * maxposs = (int *) (maxvalues + blockSize);
			//-------------
			int tid = cuGetThreadID();
			//-------------
			cudafloat min_value = MAX_CUDAFLOAT;
			cudafloat max_value = -MAX_CUDAFLOAT;
			//only work on I_low & I_high sets
			if (tid < input_size) {
				cudafloat alpha_i = alphas[tid];
				int y_i = y[tid];

				//define sets
				// 		I0 <- 0 < ai < C
				// 		I1 <- yi > 0, ai = 0
				// 		I2 <- yi < 0, ai = C
				// 		I3 <- yi > 0, ai = C
				// 		I4 <- yi < 0, ai = 0

				cudafloat constant_c_minus_eps = constant_c - constant_epsilon;

				bool I0 = (alpha_i > constant_epsilon) && (alpha_i < constant_c_minus_eps);
				bool I1 = (y_i > 0) && (alpha_i < constant_epsilon);
				bool I2 = (y_i < 0) && (alpha_i > constant_c_minus_eps);
				bool I3 = (y_i > 0) && (alpha_i > constant_c_minus_eps);
				bool I4 = (y_i < 0) && (alpha_i < constant_epsilon);

				//I_HIGH set
				if(I0||I1||I2){
					min_value = f[tid];
				}
				//I_LOW set
				if(I0||I3||I4){
					max_value = f[tid];
				}
			}
			minvalues[threadIdx.x] = min_value;
			maxvalues[threadIdx.x] = max_value;
			minposs[threadIdx.x] = tid;
			maxposs[threadIdx.x] = tid;

			__syncthreads();
			//-------------

			if (blockSize >= 1024) {
				if (threadIdx.x < 512) {
					if (minvalues[threadIdx.x] > minvalues[threadIdx.x + 512]) {
						minvalues[threadIdx.x] = minvalues[threadIdx.x + 512];
						minposs[threadIdx.x] = minposs[threadIdx.x + 512];
					}
					if (maxvalues[threadIdx.x] < maxvalues[threadIdx.x + 512]) {
						maxvalues[threadIdx.x] = maxvalues[threadIdx.x + 512];
						maxposs[threadIdx.x] = maxposs[threadIdx.x + 512];
					}
				}
				__syncthreads();
			}

			if (blockSize >= 512) {
				if (threadIdx.x < 256) {
					if (minvalues[threadIdx.x] > minvalues[threadIdx.x + 256]) {
						minvalues[threadIdx.x] = minvalues[threadIdx.x + 256];
						minposs[threadIdx.x] = minposs[threadIdx.x + 256];
					}
					if (maxvalues[threadIdx.x] < maxvalues[threadIdx.x + 256]) {
						maxvalues[threadIdx.x] = maxvalues[threadIdx.x + 256];
						maxposs[threadIdx.x] = maxposs[threadIdx.x + 256];
					}
				}
				__syncthreads();
			}

			if (blockSize >= 256) {
				if (threadIdx.x < 128) {
					if (minvalues[threadIdx.x] > minvalues[threadIdx.x + 128]) {
						minvalues[threadIdx.x] = minvalues[threadIdx.x + 128];
						minposs[threadIdx.x] = minposs[threadIdx.x + 128];
					}
					if (maxvalues[threadIdx.x] < maxvalues[threadIdx.x + 128]) {
						maxvalues[threadIdx.x] = maxvalues[threadIdx.x + 128];
						maxposs[threadIdx.x] = maxposs[threadIdx.x + 128];
					}
				}
				__syncthreads();
			}

			if (blockSize >= 128) {
				if (threadIdx.x < 64) {
					if (minvalues[threadIdx.x] > minvalues[threadIdx.x + 64]) {
						minvalues[threadIdx.x] = minvalues[threadIdx.x + 64];
						minposs[threadIdx.x] = minposs[threadIdx.x + 64];
					}
					if (maxvalues[threadIdx.x] < maxvalues[threadIdx.x + 64]) {
						maxvalues[threadIdx.x] = maxvalues[threadIdx.x + 64];
						maxposs[threadIdx.x] = maxposs[threadIdx.x + 64];
					}
				}
				__syncthreads();
			}

			if (threadIdx.x < 32) {
				volatile cudafloat * _minvalues = minvalues;
				volatile int * _minposs = minposs;
				volatile cudafloat * _maxvalues = maxvalues;
				volatile int * _maxposs = maxposs;

				if (blockSize >= 64) {
					if (_minvalues[threadIdx.x] > _minvalues[threadIdx.x + 32]) {
						_minvalues[threadIdx.x] = _minvalues[threadIdx.x + 32];
						_minposs[threadIdx.x] = _minposs[threadIdx.x + 32];
					}
					if (_maxvalues[threadIdx.x] < _maxvalues[threadIdx.x + 32]) {
						_maxvalues[threadIdx.x] = _maxvalues[threadIdx.x + 32];
						_maxposs[threadIdx.x] = _maxposs[threadIdx.x + 32];
					}
				}

				if (blockSize >= 32) {
					if (_minvalues[threadIdx.x] > _minvalues[threadIdx.x + 16]) {
						_minvalues[threadIdx.x] = _minvalues[threadIdx.x + 16];
						_minposs[threadIdx.x] = _minposs[threadIdx.x + 16];
					}
					if (_maxvalues[threadIdx.x] < _maxvalues[threadIdx.x + 16]) {
						_maxvalues[threadIdx.x] = _maxvalues[threadIdx.x + 16];
						_maxposs[threadIdx.x] = _maxposs[threadIdx.x + 16];
					}
				}

				if (blockSize >= 16) {
					if (_minvalues[threadIdx.x] > _minvalues[threadIdx.x + 8]) {
						_minvalues[threadIdx.x] = _minvalues[threadIdx.x + 8];
						_minposs[threadIdx.x] = _minposs[threadIdx.x + 8];
					}
					if (_maxvalues[threadIdx.x] < _maxvalues[threadIdx.x + 8]) {
						_maxvalues[threadIdx.x] = _maxvalues[threadIdx.x + 8];
						_maxposs[threadIdx.x] = _maxposs[threadIdx.x + 8];
					}
				}

				if (blockSize >= 8) {
					if (_minvalues[threadIdx.x] > _minvalues[threadIdx.x + 4]) {
						_minvalues[threadIdx.x] = _minvalues[threadIdx.x + 4];
						_minposs[threadIdx.x] = _minposs[threadIdx.x + 4];
					}
					if (_maxvalues[threadIdx.x] < _maxvalues[threadIdx.x + 4]) {
						_maxvalues[threadIdx.x] = _maxvalues[threadIdx.x + 4];
						_maxposs[threadIdx.x] = _maxposs[threadIdx.x + 4];
					}
				}

				if (blockSize >= 4) {
					if (_minvalues[threadIdx.x] > _minvalues[threadIdx.x + 2]) {
						_minvalues[threadIdx.x] = _minvalues[threadIdx.x + 2];
						_minposs[threadIdx.x] = _minposs[threadIdx.x + 2];
					}
					if (_maxvalues[threadIdx.x] < _maxvalues[threadIdx.x + 2]) {
						_maxvalues[threadIdx.x] = _maxvalues[threadIdx.x + 2];
						_maxposs[threadIdx.x] = _maxposs[threadIdx.x + 2];
					}
				}

				if (blockSize >= 2) {
					if (_minvalues[threadIdx.x] > _minvalues[threadIdx.x + 1]) {
						_minvalues[threadIdx.x] = _minvalues[threadIdx.x + 1];
						_minposs[threadIdx.x] = _minposs[threadIdx.x + 1];
					}
					if (_maxvalues[threadIdx.x] < _maxvalues[threadIdx.x + 1]) {
						_maxvalues[threadIdx.x] = _maxvalues[threadIdx.x + 1];
						_maxposs[threadIdx.x] = _maxposs[threadIdx.x + 1];
					}
				}

				if (threadIdx.x == 0) {
					minimuns[blockIdx.x] = minvalues[0];
					min_indices[blockIdx.x] = minposs[0];
					maximuns[blockIdx.x] = maxvalues[0];
					max_indices[blockIdx.x] = maxposs[0];
				}
			}
	}

	//just to be executed by one block! output on min_i max_i min max
	template<int blockSize> KERNEL cuFirstOrderHeuristicFinalPass(cudafloat * minimuns_input, int * min_indices_input, cudafloat * maximuns_input,
		int * max_indices_input, int input_size) {

			// local arrays (shared memory)
			extern __shared__ cudafloat
				minvalues[];
			int * minposs = (int *) (minvalues + blockDim.x);
			//-------------
			cudafloat * maxvalues = (cudafloat*) (minposs + blockDim.x);
			int * maxposs = (int *) (maxvalues + blockDim.x);
			//-------------
			minvalues[threadIdx.x] = MAX_CUDAFLOAT;
			maxvalues[threadIdx.x] = -MAX_CUDAFLOAT;
			minposs[threadIdx.x] = threadIdx.x;
			maxposs[threadIdx.x] = threadIdx.x;

			//-----------------
			for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
				if (minvalues[threadIdx.x] > minimuns_input[i]) {
					minvalues[threadIdx.x] = minimuns_input[i];
					minposs[threadIdx.x] = min_indices_input[i];
				}
				if (maxvalues[threadIdx.x] < maximuns_input[i]) {
					maxvalues[threadIdx.x] = maximuns_input[i];
					maxposs[threadIdx.x] = max_indices_input[i];
				}
			}
			__syncthreads();

			if (blockSize >= 1024) {
				if (threadIdx.x < 512) {
					if (minvalues[threadIdx.x] > minvalues[threadIdx.x + 512]) {
						minvalues[threadIdx.x] = minvalues[threadIdx.x + 512];
						minposs[threadIdx.x] = minposs[threadIdx.x + 512];
					}
					if (maxvalues[threadIdx.x] < maxvalues[threadIdx.x + 512]) {
						maxvalues[threadIdx.x] = maxvalues[threadIdx.x + 512];
						maxposs[threadIdx.x] = maxposs[threadIdx.x + 512];
					}
				}
				__syncthreads();
			}

			if (blockSize >= 512) {
				if (threadIdx.x < 256) {
					if (minvalues[threadIdx.x] > minvalues[threadIdx.x + 256]) {
						minvalues[threadIdx.x] = minvalues[threadIdx.x + 256];
						minposs[threadIdx.x] = minposs[threadIdx.x + 256];
					}
					if (maxvalues[threadIdx.x] < maxvalues[threadIdx.x + 256]) {
						maxvalues[threadIdx.x] = maxvalues[threadIdx.x + 256];
						maxposs[threadIdx.x] = maxposs[threadIdx.x + 256];
					}
				}
				__syncthreads();
			}

			if (blockSize >= 256) {
				if (threadIdx.x < 128) {
					if (minvalues[threadIdx.x] > minvalues[threadIdx.x + 128]) {
						minvalues[threadIdx.x] = minvalues[threadIdx.x + 128];
						minposs[threadIdx.x] = minposs[threadIdx.x + 128];
					}
					if (maxvalues[threadIdx.x] < maxvalues[threadIdx.x + 128]) {
						maxvalues[threadIdx.x] = maxvalues[threadIdx.x + 128];
						maxposs[threadIdx.x] = maxposs[threadIdx.x + 128];
					}
				}
				__syncthreads();
			}

			if (blockSize >= 128) {
				if (threadIdx.x < 64) {
					if (minvalues[threadIdx.x] > minvalues[threadIdx.x + 64]) {
						minvalues[threadIdx.x] = minvalues[threadIdx.x + 64];
						minposs[threadIdx.x] = minposs[threadIdx.x + 64];
					}
					if (maxvalues[threadIdx.x] < maxvalues[threadIdx.x + 64]) {
						maxvalues[threadIdx.x] = maxvalues[threadIdx.x + 64];
						maxposs[threadIdx.x] = maxposs[threadIdx.x + 64];
					}
				}
				__syncthreads();
			}

			if (threadIdx.x < 32) {
				volatile cudafloat * _minvalue = minvalues;
				volatile int * _minpos = minposs;
				volatile cudafloat * _maxvalue = maxvalues;
				volatile int * _maxpos = maxposs;

				if (blockSize >= 64) {
					if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 32]) {
						_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 32];
						_minpos[threadIdx.x] = _minpos[threadIdx.x + 32];
					}
					if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 32]) {
						_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 32];
						_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 32];
					}
				}

				if (blockSize >= 32) {
					if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 16]) {
						_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 16];
						_minpos[threadIdx.x] = _minpos[threadIdx.x + 16];
					}
					if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 16]) {
						_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 16];
						_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 16];
					}
				}

				if (blockSize >= 16) {
					if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 8]) {
						_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 8];
						_minpos[threadIdx.x] = _minpos[threadIdx.x + 8];
					}
					if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 8]) {
						_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 8];
						_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 8];
					}
				}

				if (blockSize >= 8) {
					if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 4]) {
						_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 4];
						_minpos[threadIdx.x] = _minpos[threadIdx.x + 4];
					}
					if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 4]) {
						_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 4];
						_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 4];
					}
				}

				if (blockSize >= 4) {
					if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 2]) {
						_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 2];
						_minpos[threadIdx.x] = _minpos[threadIdx.x + 2];
					}
					if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 2]) {
						_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 2];
						_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 2];
					}
				}

				if (blockSize >= 2) {
					if (_minvalue[threadIdx.x] > _minvalue[threadIdx.x + 1]) {
						_minvalue[threadIdx.x] = _minvalue[threadIdx.x + 1];
						_minpos[threadIdx.x] = _minpos[threadIdx.x + 1];
					}
					if (_maxvalue[threadIdx.x] < _maxvalue[threadIdx.x + 1]) {
						_maxvalue[threadIdx.x] = _maxvalue[threadIdx.x + 1];
						_maxpos[threadIdx.x] = _maxpos[threadIdx.x + 1];
					}
				}

				if (threadIdx.x == 0) {
					//high from min Ihigh
					d_b_high = minvalues[0];
					d_i_high = minposs[0];
					//low from max Ilow
					d_b_low = maxvalues[0];
					d_i_low = maxposs[0];
				}
			}
	}
}

#endif
