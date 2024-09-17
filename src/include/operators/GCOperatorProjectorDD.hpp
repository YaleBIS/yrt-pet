/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "operators/GCOperatorProjector.hpp"

#include <vector>

typedef std::vector<int> PositionList;

class Image;
class ProjectionData;

class GCOperatorProjectorDD : public GCOperatorProjector
{
public:
	GCOperatorProjectorDD(const GCOperatorProjectorParams& p_projParams);

	double forwardProjection(
	    const Image* in_image, const StraightLineParam& lor,
	    const Vector3D& n1, const Vector3D& n2,
	    const GCTimeOfFlightHelper* tofHelper = nullptr, float tofValue = 0.0f,
	    const GCProjectionPsfManager* psfManager = nullptr) const;

	void backProjection(Image* in_image, const StraightLineParam& lor,
	                    const Vector3D& n1, const Vector3D& n2,
	                    double proj_value,
	                    const GCTimeOfFlightHelper* tofHelper = nullptr,
	                    float tofValue = 0.0f,
	                    const GCProjectionPsfManager* psfManager = nullptr) const;

	double forwardProjection(const Image* img, const ProjectionData* dat,
	                         bin_t bin) override;

	void backProjection(Image* img, const ProjectionData* dat, bin_t bin,
	                    double projValue) override;

	static float get_overlap_safe(float p0, float p1, float d0, float d1);
	static float get_overlap_safe(float p0, float p1, float d0, float d1,
	                              const GCProjectionPsfManager* psfManager,
	                              const float* psfKernel);
	static float get_overlap(float p0, float p1, float d0, float d1,
	                         const GCProjectionPsfManager* psfManager = nullptr,
	                         const float* psfKernel = nullptr);


private:
	template <bool IS_FWD, bool FLAG_TOF>
	void dd_project_ref(Image* in_image, const StraightLineParam& lor,
	                    const Vector3D& n1, const Vector3D& n2,
	                    double& proj_value,
	                    const GCTimeOfFlightHelper* tofHelper = nullptr,
	                    float tofValue = 0.f,
	                    const GCProjectionPsfManager* psfManager = nullptr) const;
};
