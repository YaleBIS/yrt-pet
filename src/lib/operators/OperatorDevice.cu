/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/OperatorDevice.cuh"

#include "datastruct/image/Image.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "utils/GCGPUUtils.cuh"
#include "utils/Globals.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_operatorprojectordevice(py::module& m)
{
	auto c = py::class_<OperatorProjectorDevice, OperatorProjectorBase>(
	    m, "OperatorProjectorDevice");
}
#endif

namespace Util
{
	GCGPULaunchParams3D initiateDeviceParameters(const ImageParams& params)
	{
		GCGPULaunchParams3D launchParams;
		if (params.nz > 1)
		{
			const size_t threadsPerBlockDimImage =
			    GlobalsCuda::ThreadsPerBlockImg3d;
			const auto threadsPerBlockDimImage_float =
			    static_cast<float>(threadsPerBlockDimImage);
			const auto threadsPerBlockDimImage_uint =
			    static_cast<unsigned int>(threadsPerBlockDimImage);

			launchParams.gridSize = {
			    static_cast<unsigned int>(
			        std::ceil(params.nx / threadsPerBlockDimImage_float)),
			    static_cast<unsigned int>(
			        std::ceil(params.ny / threadsPerBlockDimImage_float)),
			    static_cast<unsigned int>(
			        std::ceil(params.nz / threadsPerBlockDimImage_float))};

			launchParams.blockSize = {threadsPerBlockDimImage_uint,
			                          threadsPerBlockDimImage_uint,
			                          threadsPerBlockDimImage_uint};
		}
		else
		{
			const size_t threadsPerBlockDimImage =
			    GlobalsCuda::ThreadsPerBlockImg2d;
			const auto threadsPerBlockDimImage_float =
			    static_cast<float>(threadsPerBlockDimImage);
			const auto threadsPerBlockDimImage_uint =
			    static_cast<unsigned int>(threadsPerBlockDimImage);

			launchParams.gridSize = {
			    static_cast<unsigned int>(
			        std::ceil(params.nx / threadsPerBlockDimImage_float)),
			    static_cast<unsigned int>(
			        std::ceil(params.ny / threadsPerBlockDimImage_float)),
			    1};

			launchParams.blockSize = {threadsPerBlockDimImage_uint,
			                          threadsPerBlockDimImage_uint, 1};
		}
		return launchParams;
	}

	GCGPULaunchParams initiateDeviceParameters(size_t batchSize)
	{
		GCGPULaunchParams launchParams{};
		launchParams.gridSize = static_cast<unsigned int>(
		    std::ceil(batchSize /
		              static_cast<float>(GlobalsCuda::ThreadsPerBlockData)));
		launchParams.blockSize = GlobalsCuda::ThreadsPerBlockData;
		return launchParams;
	}
}  // namespace Util

GCCUScannerParams OperatorDevice::getCUScannerParams(const Scanner& scanner)
{
	GCCUScannerParams params;
	params.crystalSize_trans = scanner.crystalSize_trans;
	params.crystalSize_z = scanner.crystalSize_z;
	params.numDets = scanner.getNumDets();
	return params;
}

GCCUImageParams
    OperatorDevice::getCUImageParams(const ImageParams& imgParams)
{
	GCCUImageParams params;

	params.voxelNumber[0] = imgParams.nx;
	params.voxelNumber[1] = imgParams.ny;
	params.voxelNumber[2] = imgParams.nz;

	params.imgLength[0] = static_cast<float>(imgParams.length_x);
	params.imgLength[1] = static_cast<float>(imgParams.length_y);
	params.imgLength[2] = static_cast<float>(imgParams.length_z);

	params.voxelSize[0] = static_cast<float>(imgParams.vx);
	params.voxelSize[1] = static_cast<float>(imgParams.vy);
	params.voxelSize[2] = static_cast<float>(imgParams.vz);

	params.offset[0] = static_cast<float>(imgParams.off_x);
	params.offset[1] = static_cast<float>(imgParams.off_y);
	params.offset[2] = static_cast<float>(imgParams.off_z);

	return params;
}

OperatorProjectorDevice::OperatorProjectorDevice(
    const OperatorProjectorParams& projParams, bool p_synchronized,
    const cudaStream_t* pp_mainStream, const cudaStream_t* pp_auxStream)
    : OperatorProjectorBase(projParams), OperatorDevice()
{
	if (projParams.tofWidth_ps > 0.f)
	{
		setupTOFHelper(projParams.tofWidth_ps, projParams.tofNumStd);
	}

	m_batchSize = 0ull;
	m_synchonized = p_synchronized;
	mp_mainStream = pp_mainStream;
	mp_auxStream = pp_auxStream;
}

unsigned int OperatorProjectorDevice::getGridSize() const
{
	return m_launchParams.gridSize;
}
unsigned int OperatorProjectorDevice::getBlockSize() const
{
	return m_launchParams.blockSize;
}

bool OperatorProjectorDevice::isSynchronized() const
{
	return m_synchonized;
}

const cudaStream_t* OperatorProjectorDevice::getMainStream() const
{
	return mp_mainStream;
}

const cudaStream_t* OperatorProjectorDevice::getAuxStream() const
{
	return mp_auxStream;
}

void OperatorProjectorDevice::setBatchSize(size_t newBatchSize)
{
	m_batchSize = newBatchSize;
	m_launchParams = Util::initiateDeviceParameters(m_batchSize);
}

ProjectionDataDeviceOwned&
    OperatorProjectorDevice::getIntermediaryProjData()
{
	ASSERT_MSG(mp_intermediaryProjData != nullptr,
	           "Projection-space GPU Intermediary buffer not initialized");
	return *mp_intermediaryProjData;
}

const ImageDevice& OperatorProjectorDevice::getAttImageDevice() const
{
	ASSERT_MSG(mp_attImageDevice != nullptr,
	           "Device attenuation image not initialized");
	return *mp_attImageDevice;
}

const ImageDevice&
    OperatorProjectorDevice::getAttImageForBackprojectionDevice() const
{
	ASSERT_MSG(mp_attImageForBackprojectionDevice != nullptr,
	           "Device attenuation image for backprojection not initialized");
	return *mp_attImageForBackprojectionDevice;
}

size_t OperatorProjectorDevice::getBatchSize() const
{
	return m_batchSize;
}

void OperatorProjectorDevice::setAttImage(const Image* attImage)
{
	OperatorProjectorBase::setAttImage(attImage);

	mp_attImageDevice = std::make_unique<ImageDeviceOwned>(
	    attImage->getParams(), getAuxStream());
	mp_attImageDevice->allocate(getAuxStream());
	mp_attImageDevice->transferToDeviceMemory(attImage, false);
}

void OperatorProjectorDevice::setAttImageForBackprojection(
    const Image* attImage)
{
	OperatorProjectorBase::setAttImageForBackprojection(attImage);

	mp_attImageForBackprojectionDevice = std::make_unique<ImageDeviceOwned>(
	    attImage->getParams(), getAuxStream());
	mp_attImageForBackprojectionDevice->allocate(getAuxStream());
	mp_attImageForBackprojectionDevice->transferToDeviceMemory(attImage, false);
}

void OperatorProjectorDevice::setAddHisto(const Histogram* p_addHisto)
{
	OperatorProjectorBase::setAddHisto(p_addHisto);
}

void OperatorProjectorDevice::setupTOFHelper(float tofWidth_ps, int tofNumStd)
{
	mp_tofHelper = std::make_unique<GCDeviceObject<TimeOfFlightHelper>>(
	    tofWidth_ps, tofNumStd);
}

bool OperatorProjectorDevice::requiresIntermediaryProjData() const
{
	// We need an intermediary projectorParam if we'll need to do attenuation
	// correction or additive correction (scatter/randoms)
	return attImage != nullptr || attImageForBackprojection != nullptr ||
	       addHisto != nullptr;
}

void OperatorProjectorDevice::prepareIntermediaryBufferIfNeeded(
    const ProjectionDataDevice* orig)
{
	if (requiresIntermediaryProjData())
	{
		prepareIntermediaryBuffer(orig);
	}
}

void OperatorProjectorDevice::prepareIntermediaryBuffer(
    const ProjectionDataDevice* orig)
{
	if (mp_intermediaryProjData == nullptr)
	{
		mp_intermediaryProjData =
		    std::make_unique<ProjectionDataDeviceOwned>(orig);
	}
	mp_intermediaryProjData->allocateForProjValues(getAuxStream());
}

const TimeOfFlightHelper*
	OperatorProjectorDevice::getTOFHelperDevicePointer() const
{
	if(mp_tofHelper != nullptr)
	{
		return mp_tofHelper->getDevicePointer();
	}
	return nullptr;
}