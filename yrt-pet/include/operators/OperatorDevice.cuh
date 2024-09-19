/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/ImageBase.hpp"
#include "datastruct/image/ImageDevice.cuh"
#include "datastruct/projection/ProjectionDataDevice.cuh"
#include "operators/Operator.hpp"
#include "operators/OperatorProjector.hpp"
#include "recon/CUParameters.hpp"
#include "utils/DeviceObject.cuh"
#include "utils/GPUTypes.cuh"

#include <cuda_runtime_api.h>

namespace Util
{
	// Takes a reference of an image, and automatically determines
	// the 3D grid/block parameters.
	GPULaunchParams3D initiateDeviceParameters(const ImageParams& params);
	// Takes a size of the data to be processed, and automatically determines
	// the grid/block parameters
	GPULaunchParams initiateDeviceParameters(size_t batchSize);
}  // namespace Util

class OperatorDevice : public Operator
{
	// Nothing here for now...

public:
	static CUScannerParams getCUScannerParams(const Scanner& scanner);
	static CUImageParams getCUImageParams(const ImageParams& imgParams);

protected:
	OperatorDevice() = default;
};

class OperatorProjectorDevice : public OperatorProjectorBase,
                                  public OperatorDevice
{
public:
	OperatorProjectorDevice() = delete;

	size_t getBatchSize() const;

	unsigned int getGridSize() const;
	unsigned int getBlockSize() const;
	bool isSynchronized() const;

	bool requiresIntermediaryProjData() const;

	void setAttImage(const Image* attImage) override;
	void setAttImageForBackprojection(const Image* attImage) override;
	void setAddHisto(const Histogram *p_addHisto) override;
	void setupTOFHelper(float tofWidth_ps, int tofNumStd = -1);

protected:
	OperatorProjectorDevice(const OperatorProjectorParams& projParams,
	                          bool p_synchronized = true,
	                          const cudaStream_t* pp_mainStream = nullptr,
	                          const cudaStream_t* pp_auxStream = nullptr);

	const cudaStream_t* getMainStream() const;
	const cudaStream_t* getAuxStream() const;

	void setBatchSize(size_t newBatchSize);

	ProjectionDataDeviceOwned& getIntermediaryProjData();
	const ImageDevice& getAttImageDevice() const;
	const ImageDevice& getAttImageForBackprojectionDevice() const;
	void prepareIntermediaryBuffer(const ProjectionDataDevice* orig);
	void prepareIntermediaryBufferIfNeeded(const ProjectionDataDevice* orig);

	const TimeOfFlightHelper* getTOFHelperDevicePointer() const;

private:
	size_t m_batchSize;
	GPULaunchParams m_launchParams{};
	const cudaStream_t* mp_mainStream;
	const cudaStream_t* mp_auxStream;
	bool m_synchonized;

	std::unique_ptr<DeviceObject<TimeOfFlightHelper>> mp_tofHelper;

	// For attenuation correction
	std::unique_ptr<ImageDeviceOwned> mp_attImageDevice;
	// For attenuation correction
	std::unique_ptr<ImageDeviceOwned> mp_attImageForBackprojectionDevice;
	// For Attenuation correction or Additive correction
	std::unique_ptr<ProjectionDataDeviceOwned> mp_intermediaryProjData;
};