/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/GCGPUUtils.cuh"

bool cudaCheckError()
{
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != 0)
	{
		std::cerr << "CUDA Error: " << cudaGetErrorString(cudaError)
		          << std::endl;
		return false;
	}
	return true;
}

size_t getDeviceInfo(bool verbose)
{
	int devicesNb = 0;
	cudaGetDeviceCount(&devicesNb);
	cudaCheckError();
	std::cout << "\n"
	          << "*** GPUs INFORMATION ***"
	          << "\n"
	          << std::endl;
	std::cout << "Number of devices detected: " << devicesNb << std::endl;
	size_t freeMem, totalMem;
	int gpu_id_toUse = 0;
	size_t maxDeviceMem = 0;
	for (int d_id = 0; d_id < devicesNb; d_id++)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, d_id);
		cudaSetDevice(d_id);
		cudaMemGetInfo(&freeMem, &totalMem);
		if (verbose)
		{
			std::cout << "Device name: " << deviceProp.name << std::endl;
			std::cout << "Compute capability: " << deviceProp.major << "."
			          << deviceProp.minor << std::endl;
			std::cout << "Number of asynchronous engines: "
			          << deviceProp.asyncEngineCount << std::endl;
			std::cout << "Device memory - Total memory: "
			          << totalMem / (1024 * 1024 * 1024)
			          << "GiB - Available memory: "
			          << freeMem / (1024 * 1024 * 1024) << "GiB \n"
			          << std::endl;
		}
		if (freeMem > maxDeviceMem)
		{
			maxDeviceMem = freeMem;
			gpu_id_toUse = d_id;
		}
	}
	std::cout << "Selected device id: " << gpu_id_toUse << "\n" << std::endl;
	cudaSetDevice(gpu_id_toUse);
	return maxDeviceMem;
}

void gpuAssert(cudaError_t code, const char* file, int line, bool abort)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
		        line);
		if (abort)
			exit(code);
	}
}
