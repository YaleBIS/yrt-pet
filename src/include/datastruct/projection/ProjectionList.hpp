/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/projection/ProjectionData.hpp"
#include "utils/Array.hpp"

#include <memory>


// A Class that stores projection values, but does not
// store the lines associated to each one.
// Instead, it uses a reference projection class to get the line.
// Useful for temporary projections where you don't want to necessarily
// copy all the detector coordinates in your list-mode/histogram.

class ProjectionList : public ProjectionData
{
public:
	ProjectionList(const ProjectionData* r);
	~ProjectionList() override = default;

	size_t count() const override;
	float getProjectionValue(bin_t id) const override;
	void setProjectionValue(bin_t id, float val) override;
	det_id_t getDetector1(bin_t evId) const override;
	det_id_t getDetector2(bin_t evId) const override;
	det_pair_t getDetectorPair(bin_t evId) const override;
	histo_bin_t getHistogramBin(bin_t id) const override;
	std::unique_ptr<BinIterator> getBinIter(int numSubsets,
	                                          int idxSubset) const override;
	frame_t getFrame(bin_t id) const override;
	timestamp_t getTimestamp(bin_t id) const override;
	size_t getNumFrames() const override;
	bool isUniform() const override;
	float getRandomsEstimate(bin_t id) const override;
	bool hasTOF() const override;
	float getTOFValue(bin_t id) const override;
	bool hasMotion() const override;
	transform_t getTransformOfFrame(frame_t frame) const override;
	bool hasArbitraryLORs() const override;
	line_t getArbitraryLOR(bin_t id) const override;

	const ProjectionData* getReference() const { return mp_reference; }

	void clearProjections(float value) override;
	Array1DBase<float>* getProjectionsArrayRef() const;

protected:
	const ProjectionData* mp_reference;
	std::unique_ptr<Array1DBase<float>> mp_projs;
};

class ProjectionListAlias : public ProjectionList
{
public:
	ProjectionListAlias(ProjectionData* p);
	~ProjectionListAlias() override = default;
	void bind(Array1DBase<float>* projs_in);
};

class ProjectionListOwned : public ProjectionList
{
public:
	ProjectionListOwned(ProjectionData* p);
	~ProjectionListOwned() override = default;
	void allocate();
};
