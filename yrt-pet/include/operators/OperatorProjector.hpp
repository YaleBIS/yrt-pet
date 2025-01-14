/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/projection/ProjectionData.hpp"
#include "operators/OperatorProjectorBase.hpp"
#include "operators/Operator.hpp"
#include "operators/ProjectionPsfManager.hpp"
#include "operators/TimeOfFlight.hpp"
#include "utils/Types.hpp"

class BinIterator;
class Image;
class Scanner;
class ProjectionData;
class Histogram;


class OperatorProjector : public OperatorProjectorBase
{
public:
	enum ProjectorType
	{
		SIDDON = 0,
		DD,
		DD_GPU
	};

	explicit OperatorProjector(const OperatorProjectorParams& p_projParams);

	// Virtual functions
	virtual float forwardProjection(
	    const Image* image,
	    const ProjectionProperties& projectionProperties) const = 0;
	virtual void
	    backProjection(Image* image,
	                   const ProjectionProperties& projectionProperties,
	                   float projValue) const = 0;

	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;

	void setupTOFHelper(float tofWidth_ps, int tofNumStd = -1);
	void setupProjPsfManager(const std::string& psfFilename);

	const TimeOfFlightHelper* getTOFHelper() const;
	const ProjectionPsfManager* getProjectionPsfManager() const;

protected:
	// Time of flight
	std::unique_ptr<TimeOfFlightHelper> mp_tofHelper;

	// Projection-domain PSF
	std::unique_ptr<ProjectionPsfManager> mp_projPsfManager;
};
