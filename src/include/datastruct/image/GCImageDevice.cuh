/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/GCImageBase.hpp"
#include "utils/GCGPUTypes.cuh"
#include "utils/GCPageLockedBuffer.cuh"

#include <cuda_runtime_api.h>

class GCImage;

class GCImageDevice : public GCImageBase
{
public:
	GCImageDevice(const GCImageParams& imgParams,
	              const cudaStream_t* stream_ptr = nullptr);

	virtual float* getDevicePointer() = 0;
	virtual const float* getDevicePointer() const = 0;
	size_t getImageSize() const;
	const cudaStream_t* getStream() const;
	void transferToDeviceMemory(const float* hp_img_ptr,
	                            bool p_synchronize = false);
	void transferToDeviceMemory(const GCImage* hp_img_ptr,
	                            bool p_synchronize = false);
	void transferToHostMemory(float* hp_img_ptr,
	                          bool p_synchronize = false) const;
	void transferToHostMemory(GCImage* hp_img_ptr,
	                          bool p_synchronize = false) const;
	void setValue(double initValue) override;
	void addFirstImageToSecond(GCImageBase* imgOut) const override;
	void applyThreshold(const GCImageBase* maskImg, double threshold,
	                    double val_le_scale, double val_le_off,
	                    double val_gt_scale, double val_gt_off) override;
	void updateEMThreshold(GCImageBase* updateImg, const GCImageBase* normImg,
	                       double threshold) override;
	void writeToFile(const std::string& image_fname) const override;

	void applyThresholdDevice(const GCImageDevice* maskImg, float threshold,
	                          float val_le_scale, float val_le_off,
	                          float val_gt_scale, float val_gt_off);

protected:
	size_t m_imgSize;
	const cudaStream_t* mp_stream;

private:
	GCGPULaunchParams3D m_launchParams;

	// For Host->Device data transfers
	mutable GCPageLockedBuffer<float> m_tempBuffer;
};

class GCImageDeviceOwned : public GCImageDevice
{
public:
	GCImageDeviceOwned(const GCImageParams& imgParams,
	                   const cudaStream_t* stream_ptr = nullptr);
	GCImageDeviceOwned(const GCImage* img_ptr,
	                   const cudaStream_t* stream_ptr = nullptr);
	GCImageDeviceOwned(const GCImageParams& imgParams,
	                   const std::string& filename,
	                   const cudaStream_t* stream_ptr = nullptr);
	~GCImageDeviceOwned() override;
	void allocate(bool synchronize = true);
	void readFromFile(const std::string& filename);
	float* getDevicePointer() override;
	const float* getDevicePointer() const override;

private:
	float* mpd_devicePointer;  // Device data
};

class GCImageDeviceAlias : public GCImageDevice
{
public:
	GCImageDeviceAlias(const GCImageParams& imgParams,
			   const cudaStream_t* stream_ptr = nullptr);
	float* getDevicePointer() override;
	const float* getDevicePointer() const override;
	size_t getDevicePointerInULL() const;

	void setDevicePointer(float* ppd_devicePointer);
	void setDevicePointer(size_t ppd_pointerInULL);
	bool isDevicePointerSet() const;

private:
	float* mpd_devicePointer;
};
