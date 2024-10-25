/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/ImageBase.hpp"
#include "geometry/Vector3D.hpp"
#include "utils/Array.hpp"

#include <sitkImage.h>

#include <string>

class Image : public ImageBase
{
public:
	~Image() override = default;

	Array3DBase<float>& getData();
	const Array3DBase<float>& getData() const;
	float* getRawPointer();
	const float* getRawPointer() const;
	bool isMemoryValid() const;

	void copyFromImage(const ImageBase* imSrc) override;
	void multWithScalar(float scalar);
	void addFirstImageToSecond(ImageBase* secondImage) const override;

	void setValue(float initValue) override;
	void applyThreshold(const ImageBase* maskImg, float threshold,
	                    float val_le_scale, float val_le_off,
	                    float val_gt_scale, float val_gt_off) override;
	void updateEMThreshold(ImageBase* updateImg, const ImageBase* normImg,
	                       float threshold) override;
	void writeToFile(const std::string& fname) const override;

	Array3DAlias<float> getArray() const;
	std::unique_ptr<Image> transformImage(const Vector3D& rotation,
	                                      const Vector3D& translation) const;

	float dotProduct(const Image& y) const;
	float nearestNeighbor(const Vector3D& pt) const;
	float nearestNeighbor(const Vector3D& pt, int* pi, int* pj, int* pk) const;
	void updateImageNearestNeighbor(const Vector3D& pt, float value,
	                                bool mult_flag);
	void assignImageNearestNeighbor(const Vector3D& pt, float value);
	bool getNearestNeighborIdx(const Vector3D& pt, int* pi, int* pj,
	                           int* pk) const;

	float interpolateImage(const Vector3D& pt) const;
	float interpolateImage(const Vector3D& pt, const Image& sens) const;
	void updateImageInterpolate(const Vector3D& point, float value,
	                            bool mult_flag);
	void assignImageInterpolate(const Vector3D& point, float value);

protected:
	static ImageParams
	    createImageParamsFromSitkImage(const itk::simple::Image& sitkImage);
	static float sitkOriginToImageParamsOffset(double sitkOrigin,
	                                           float voxelSize, float length);
	static double imageParamsOffsetToSitkOrigin(float off, float voxelSize,
	                                            float length);
	static void updateSitkImageFromParameters(itk::simple::Image& sitkImage,
					  const ImageParams& params);

	Image();
	explicit Image(const ImageParams& imgParams);
	std::unique_ptr<Array3DBase<float>> mp_array;

};

class ImageOwned : public Image
{
public:
	explicit ImageOwned(const ImageParams& imgParams);
	ImageOwned(const ImageParams& imgParams, const std::string& filename);
	explicit ImageOwned(const std::string& filename);
	void allocate();
	void readFromFile(const std::string& fname);
	void writeToFile(const std::string& fname) const override;

private:
	void checkImageParamsWithSitkImage() const;
	std::unique_ptr<itk::simple::Image> mp_sitkImage;
};

class ImageAlias : public Image
{
public:
	explicit ImageAlias(const ImageParams& imgParams);
	void bind(Array3DBase<float>& p_data);
};
