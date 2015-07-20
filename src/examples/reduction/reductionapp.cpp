#include <stdio.h>
#include <stdlib.h>
#include <ctime>

#include "../../reduction/reduction.h"
#include "../common/CudaInit.h"

using namespace GPUMLib;

#define TOLERANCE (0.00001)

cudafloat absdiff(cudafloat value, cudafloat expected) {
	cudafloat diff = value - expected;

	if (diff < CUDA_VALUE(0.0)) return -diff;
	return diff;
}

void Display(const char * name, cudafloat value, cudafloat expected) {
	cout << name << ": " << value << " ";
	if (absdiff(value, expected) < TOLERANCE) {
		cout << "ok";
	} else {
		cout << "error (expected ";
		cout << expected << ")";
	}
	cout << endl;
}

int main(int argc, char * argv[])
{
	CudaDevice device;
	if(!device.SupportsCuda()) {
		cout << "Device does not support cuda" << endl;
		return 0;
	}

	if (argc != 3 && argc != 4) {
		cout << "usage: reduction <array size> <number tests> [<random generator]" << endl;
		return 0;
	}

	int N = atoi(argv[1]);
	int tests = atoi(argv[2]);

	unsigned randomGenerator = (argc == 4) ? atol(argv[3]) : (unsigned int) time(0);
	srand(randomGenerator);

	device.ShowInfo();
	cout << "Random Generator.........: " << randomGenerator << endl;

	CudaArray<cudafloat> x;
	CudaArray<cudafloat> min(1);
	CudaArray<cudafloat> max(1);
	CudaArray<cudafloat> sum(1);
	CudaArray<cudafloat> avg(1);
	CudaArray<int> minIdx(1);
	CudaArray<int> maxIdx(1);

	x.ResizeWithoutPreservingData(N);

	for(int i = 0; i < N; i++) x[i] = CUDA_VALUE(-1.0) + (CUDA_VALUE(2.0) * rand()) / RAND_MAX;
	x.UpdateDevice();

	clock_t initialTime = clock();

	for(int t = 0; t < tests; t++) {
		Reduction::Sum(x.GetDeviceArray(), sum.GetDeviceArray());
		Reduction::Average(x.GetDeviceArray(), avg.GetDeviceArray());
		Reduction::MinIndex(x.GetDeviceArray(), min.GetDeviceArray(), minIdx.GetDeviceArray());
		Reduction::MaxIndex(x.GetDeviceArray(), max.GetDeviceArray(), maxIdx.GetDeviceArray());
		//Reduction::Min(x.GetDeviceArray(), min.GetDeviceArray());
		//Reduction::Max(x.GetDeviceArray(), max.GetDeviceArray());
	}

	cudaThreadSynchronize();
	unsigned time = (clock() - initialTime);
	
	sum.UpdateHost();
	avg.UpdateHost();
	min.UpdateHost();
	max.UpdateHost();
	minIdx.UpdateHost();
	maxIdx.UpdateHost();

	cudafloat hsum = x[0];
	cudafloat hmin = x[0];
	cudafloat hmax = x[0];
	int hposmin = 0;
	int hposmax = 0;
	
	for(int i = 1; i < N; i++) {
		if (x[i] < hmin) {
			hmin = x[i];
			hposmin = i;
		} else if (x[i] > hmax) {
			hmax = x[i];
			hposmax = i;
		}
		hsum += x[i];
	}

	Display("sum", sum[0], hsum);
	Display("avg", avg[0], hsum / N);
	Display("min", min[0], hmin);
	Display("max", max[0], hmax);
	Display("min index", (cudafloat) minIdx[0], (cudafloat) hposmin);
	if (minIdx[0] != hposmin) cout << "x[" << minIdx[0] << "] = " << x[minIdx[0]] << "; x["<< hposmin << "] = " << x[hposmin] << endl;
	Display("max index", (cudafloat) maxIdx[0], (cudafloat) hposmax);
	if (maxIdx[0] != hposmax) cout << "x[" << maxIdx[0] << "] = " << x[maxIdx[0]] << "; x["<< hposmax << "] = " << x[hposmax] << endl;

	cout << (double) time / CLOCKS_PER_SEC << " s" << endl;

	return 0;
}
