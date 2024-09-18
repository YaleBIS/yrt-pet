/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/ImageBase.hpp"
#include "utils/GPUTypes.cuh"
#include "utils/PageLockedBuffer.cuh"

#include <cuda_runtime_api.h>

class Image;

class ImageDevice : public ImageBase
{
public:
	ImageDevice(const ImageParams& imgParams,
	              const cudaStream_t* stream_ptr = nullptr);

	virtual float* getDevicePointer() = 0;
	virtual const float* getDevicePointer() const = 0;
	size_t getImageSize() const;
	const cudaStream_t* getStream() const;
	void transferToDeviceMemory(const float* hp_img_ptr,
	                            bool p_synchronize = false);
	void transferToDeviceMemory(const Image* hp_img_ptr,
	                            bool p_synchronize = false);
	void transferToHostMemory(float* hp_img_ptr,
	                          bool p_synchronize = false) const;
	void transferToHostMemory(Image* hp_img_ptr,
	                          bool p_synchronize = false) const;
	void setValue(double initValue) override;
	void addFirstImageToSecond(ImageBase* imgOut) const override;
	void applyThreshold(const ImageBase* maskImg, double threshold,
	                    double val_le_scale, double val_le_off,
	                    double val_gt_scale, double val_gt_off) override;
	void updateEMThreshold(ImageBase* updateImg, const ImageBase* normImg,
	                       double threshold) override;
	void writeToFile(const std::string& image_fname) const override;

	void applyThresholdDevice(const ImageDevice* maskImg, float threshold,
	                          float val_le_scale, float val_le_off,
	                          float val_gt_scale, float val_gt_off);

protected:
	size_t m_imgSize;
	const cudaStream_t* mp_stream;

private:
	GPULaunchParams3D m_launchParams;

	// For Host->Device data transfers
	mutable PageLockedBuffer<float> m_tempBuffer;
};

class ImageDeviceOwned : public ImageDevice
{
public:
	ImageDeviceOwned(const ImageParams& imgParams,
	                   const cudaStream_t* stream_ptr = nullptr);
	ImageDeviceOwned(const Image* img_ptr,
	                   const cudaStream_t* stream_ptr = nullptr);
	ImageDeviceOwned(const ImageParams& imgParams,
	                   const std::string& filename,
	                   const cudaStream_t* stream_ptr = nullptr);
	~ImageDeviceOwned() override;
	void allocate(bool synchronize = true);
	void readFromFile(const std::string& filename);
	float* getDevicePointer() override;
	const float* getDevicePointer() const override;

private:
	float* mpd_devicePointer;  // Device data
};

class ImageDeviceAlias : public ImageDevice
{
public:
	ImageDeviceAlias(const ImageParams& imgParams,
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
