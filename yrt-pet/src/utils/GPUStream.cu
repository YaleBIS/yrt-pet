/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/GPUStream.cuh"

GPUStream::GPUStream(unsigned int flags)
{
	cudaStreamCreateWithFlags(&m_stream, flags);
}

const cudaStream_t& GPUStream::getStream() const
{
	return m_stream;
}

GPUStream::~GPUStream()
{
	cudaStreamDestroy(m_stream);
}