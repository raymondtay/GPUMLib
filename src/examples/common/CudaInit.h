#ifndef CudaInit_h
#define CudaInit_h

#include <cuda_runtime.h>
#include "../../common/CudaDefinitions.h"

#ifdef _CONSOLE
	#include <iostream>
	using namespace std;
#endif

class CudaDevice {
	private:
		#if !__DEVICE_EMULATION__

		cudaDeviceProp deviceProperties;
		bool deviceSuportsCuda;
		int device;

		#endif

		#ifdef _CONSOLE

			void ShowProperty(const char * name, size_t value) {
				cout << name << value << endl;
			}

		#endif

	public:
		#if __DEVICE_EMULATION__

			bool SupportsCuda() {
				return true;
			}

			int MaxThreadsPerBlock() {
				return 512;
			}

			#ifdef _CONSOLE

				void ShowInfo() {
					cout << "device...................: Emulation" << endl;
					ShowProperty("Size of floating type....: ", sizeof(cudafloat));
				}

			#endif

		#else

			CudaDevice() {
				deviceSuportsCuda = false;

				int numberDevices;
				if (cudaGetDeviceCount(&numberDevices) != cudaSuccess) return;
				
				for(device = 0; device < numberDevices; device++) {
					if(cudaGetDeviceProperties(&deviceProperties, device) == cudaSuccess && deviceProperties.major >= 1) {
						if (cudaSetDevice(device) == cudaSuccess) {
							deviceSuportsCuda = true;
							return;
						}
					}
				}
			}

			bool SupportsCuda() {
				return deviceSuportsCuda;
			}

			int MaxThreadsPerBlock() {
				return deviceProperties.maxThreadsPerBlock;
			}

			#ifdef _CONSOLE

				void ShowInfo() {
					ShowProperty("device...................: ", device);

					cout << "Name.....................: " << deviceProperties.name << " [" << (deviceProperties.clockRate/1000) << "Mhz - supports CUDA " << deviceProperties.major << "." << deviceProperties.minor << endl;

					ShowProperty("Multi-Processors.........: ", deviceProperties.multiProcessorCount);
					ShowProperty("Global mem...............: ", deviceProperties.totalGlobalMem);
					ShowProperty("Const mem................: ", deviceProperties.totalConstMem);
					ShowProperty("Shared mem per block.....: ", deviceProperties.sharedMemPerBlock);
					ShowProperty("Regs per block...........: ", deviceProperties.regsPerBlock);
					ShowProperty("Max threads per block....: ", deviceProperties.maxThreadsPerBlock);

					cout << "Max threads dim..........: (" << deviceProperties.maxThreadsDim[0] << ", " << deviceProperties.maxThreadsDim[1] << ", " << deviceProperties.maxThreadsDim[2] << ")" << endl;
					cout << "Max grid size............: (" << deviceProperties.maxGridSize[0] << ", " << deviceProperties.maxGridSize[1] << ", " << deviceProperties.maxGridSize[2] << ")" << endl;

					ShowProperty("Warp size................: ", deviceProperties.warpSize);
					ShowProperty("Mem pitch................: ", deviceProperties.memPitch);
					ShowProperty("Texture Alignment........: ", deviceProperties.textureAlignment);
					ShowProperty("Device overlap...........: ", deviceProperties.deviceOverlap);
					ShowProperty("kernel Timeout Enabled...: ", deviceProperties.kernelExecTimeoutEnabled);
					ShowProperty("Device integrated........: ", deviceProperties.integrated);
					ShowProperty("Can map host memory......: ", deviceProperties.canMapHostMemory);
					ShowProperty("Compute mode.............: ", deviceProperties.computeMode);

					ShowProperty("Size of floating type....: ", sizeof(cudafloat));
				}

			#endif
		#endif
};

#endif
