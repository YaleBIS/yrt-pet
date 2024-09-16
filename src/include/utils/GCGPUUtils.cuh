/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#if BUILD_CUDA

#include <iostream>

#include <cuda_runtime.h>

// returns false if there is an error
__host__ bool cudaCheckError();

__host__ size_t getDeviceInfo(bool verbose = false);
void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true);

#define gpuErrchk(ans)                        \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}

#endif

#if BUILD_CUDA
#define HOST_DEVICE_CALLABLE __host__ __device__
#else
#define HOST_DEVICE_CALLABLE
#endif
