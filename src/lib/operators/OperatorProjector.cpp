/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/OperatorProjector.hpp"

#include "datastruct/image/Image.hpp"
#include "datastruct/projection/BinIterator.hpp"
#include "datastruct/projection/Histogram3D.hpp"
#include "geometry/Constants.hpp"
#include "utils/GCAssert.hpp"
#include "utils/Globals.hpp"
#include "utils/GCReconstructionUtils.hpp"
#include "utils/GCTools.hpp"

#include "omp.h"


#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

#include <utility>
namespace py = pybind11;

void py_setup_operator(py::module& m)
{
	auto c = py::class_<Operator>(m, "Operator");
}

void py_setup_operatorprojectorparams(py::module& m)
{
	auto c =
	    py::class_<OperatorProjectorParams>(m, "OperatorProjectorParams");
	c.def(py::init<BinIterator*, Scanner*, float, int, const std::string&,
	               int>(),
	      py::arg("binIter"), py::arg("scanner"), py::arg("tofWidth_ps") = 0.f,
	      py::arg("tofNumStd") = 0, py::arg("psfProjFilename") = "",
	      py::arg("num_rays") = 1);
	c.def_readwrite("tofWidth_ps", &OperatorProjectorParams::tofWidth_ps);
	c.def_readwrite("tofNumStd", &OperatorProjectorParams::tofNumStd);
	c.def_readwrite("psfProjFilename",
	                &OperatorProjectorParams::psfProjFilename);
	c.def_readwrite("num_rays", &OperatorProjectorParams::numRays);
}

void py_setup_operatorprojectorbase(py::module& m)
{
	auto c = py::class_<OperatorProjectorBase, Operator>(
	    m, "OperatorProjectorBase");
	c.def("setAddHisto", &OperatorProjectorBase::setAddHisto);
	c.def("setAttenuationImage", &OperatorProjectorBase::setAttenuationImage);
	c.def("setAttImage", &OperatorProjectorBase::setAttenuationImage);
	c.def("setAttImageForBackprojection",
	      &OperatorProjectorBase::setAttImageForBackprojection);
	c.def("getBinIter", &OperatorProjectorBase::getBinIter);
	c.def("getScanner", &OperatorProjectorBase::getScanner);
	c.def("getAttImage", &OperatorProjectorBase::getAttImage);
}

void py_setup_operatorprojector(py::module& m)
{
	auto c = py::class_<OperatorProjector, OperatorProjectorBase>(
	    m, "OperatorProjector");
	c.def("setupTOFHelper", &OperatorProjector::setupTOFHelper);
	c.def("getTOFHelper", &OperatorProjector::getTOFHelper);
	c.def("getProjectionPsfManager",
	      &OperatorProjector::getProjectionPsfManager);
	c.def(
	    "applyA",
	    [](OperatorProjector& self, const Image* img, ProjectionData* proj)
	    { self.applyA(img, proj); }, py::arg("img"), py::arg("proj"));
	c.def(
	    "applyAH",
	    [](OperatorProjector& self, const ProjectionData* proj, Image* img)
	    { self.applyAH(proj, img); }, py::arg("proj"), py::arg("img"));

	py::enum_<OperatorProjector::ProjectorType>(c, "ProjectorType")
	    .value("SIDDON", OperatorProjector::ProjectorType::SIDDON)
	    .value("DD", OperatorProjector::ProjectorType::DD)
	    .value("DD_GPU", OperatorProjector::ProjectorType::DD_GPU)
	    .export_values();
}

#endif

OperatorProjectorParams::OperatorProjectorParams(
    const BinIterator* p_binIter, const Scanner* p_scanner,
    float p_tofWidth_ps, int p_tofNumStd, std::string p_psfProjFilename,
    int p_num_rays)
    : binIter(p_binIter),
      scanner(p_scanner),
      tofWidth_ps(p_tofWidth_ps),
      tofNumStd(p_tofNumStd),
      psfProjFilename(std::move(p_psfProjFilename)),
      numRays(p_num_rays)
{
}

OperatorProjectorBase::OperatorProjectorBase(
    const OperatorProjectorParams& p_projParams)
    : binIter(p_projParams.binIter),
      scanner(p_projParams.scanner),
      attImage(nullptr),
      attImageForBackprojection(nullptr),
      addHisto(nullptr)
{
}

void OperatorProjectorBase::setAddHisto(const Histogram* p_addHisto)
{
	ASSERT_MSG(p_addHisto != nullptr,
	           "The additive histogram given in "
	           "OperatorProjector::setAddHisto is a null pointer");
	addHisto = p_addHisto;
}

void OperatorProjectorBase::setBinIter(const BinIterator* p_binIter)
{
	binIter = p_binIter;
}

void OperatorProjectorBase::setAttenuationImage(const Image* p_attImage)
{
	setAttImage(p_attImage);
}

void OperatorProjectorBase::setAttImageForBackprojection(
    const Image* p_attImage)
{
	attImageForBackprojection = p_attImage;
}

void OperatorProjectorBase::setAttImage(const Image* p_attImage)
{
	ASSERT_MSG(p_attImage != nullptr,
	           "The attenuation image given in "
	           "OperatorProjector::setAttenuationImage is a null pointer");
	attImage = p_attImage;
}

const BinIterator* OperatorProjectorBase::getBinIter() const
{
	return binIter;
}

const Scanner* OperatorProjectorBase::getScanner() const
{
	return scanner;
}

const Image* OperatorProjectorBase::getAttImage() const
{
	return attImage;
}

const Image* OperatorProjectorBase::getAttImageForBackprojection() const
{
	return attImageForBackprojection;
}

const Histogram* OperatorProjectorBase::getAddHisto() const
{
	return addHisto;
}

OperatorProjector::OperatorProjector(
    const OperatorProjectorParams& p_projParams)
    : OperatorProjectorBase(p_projParams),
      mp_tofHelper(nullptr),
      mp_projPsfManager(nullptr)
{
	if (p_projParams.tofWidth_ps > 0.f)
	{
		setupTOFHelper(p_projParams.tofWidth_ps, p_projParams.tofNumStd);
	}
	if (!p_projParams.psfProjFilename.empty())
	{
		setupProjPsfManager(p_projParams.psfProjFilename);
	}
	ASSERT_MSG_WARNING(
	    mp_projPsfManager == nullptr,
	    "Siddon does not support Projection space PSF. It will be ignored.");
}

void OperatorProjector::applyA(const GCVariable* in, GCVariable* out)
{
	auto* dat = dynamic_cast<ProjectionData*>(out);
	auto* img = dynamic_cast<const Image*>(in);

	ASSERT_MSG(dat != nullptr, "Input variable has to be Projection data");
	ASSERT_MSG(img != nullptr, "Output variable has to be an image");
#pragma omp parallel for
	for (bin_t binIdx = 0; binIdx < binIter->size(); binIdx++)
	{
		const bin_t bin = binIter->get(binIdx);

		double imProj = 0.f;
		imProj += forwardProjection(img, dat, bin);

		if (attImage != nullptr)
		{
			// Multiplicative attenuation correction (for motion)
			const double attProj = forwardProjection(attImage, dat, bin);
			const double attProj_coeff =
			    Util::getAttenuationCoefficientFactor(attProj);
			imProj *= attProj_coeff;
		}

		if (addHisto != nullptr)
		{
			// Additive correction(s)
			const histo_bin_t histoBin = dat->getHistogramBin(bin);
			imProj += addHisto->getProjectionValueFromHistogramBin(histoBin);
		}
		dat->setProjectionValue(bin, static_cast<float>(imProj));
	}
}

void OperatorProjector::applyAH(const GCVariable* in, GCVariable* out)
{
	auto* dat = dynamic_cast<const ProjectionData*>(in);
	auto* img = dynamic_cast<Image*>(out);
	ASSERT(dat != nullptr && img != nullptr);

#pragma omp parallel for
	for (bin_t binIdx = 0; binIdx < binIter->size(); binIdx++)
	{
		const bin_t bin = binIter->get(binIdx);

		double projValue = dat->getProjectionValue(bin);
		if (std::abs(projValue) < SMALL)
		{
			continue;
		}

		if (attImageForBackprojection != nullptr)
		{
			// Multiplicative attenuation correction
			const double attProj =
			    forwardProjection(attImageForBackprojection, dat, bin);
			const double attProj_coeff =
			    Util::getAttenuationCoefficientFactor(attProj);
			projValue *= attProj_coeff;
		}

		backProjection(img, dat, bin, projValue);
	}
}

void OperatorProjector::setupTOFHelper(float tofWidth_ps, int tofNumStd)
{
	mp_tofHelper =
	    std::make_unique<TimeOfFlightHelper>(tofWidth_ps, tofNumStd);
	ASSERT_MSG(mp_tofHelper != nullptr,
	           "Error occured during the setup of TimeOfFlightHelper");
}

void OperatorProjector::setupProjPsfManager(const std::string& psfFilename)
{
	mp_projPsfManager = std::make_unique<ProjectionPsfManager>(psfFilename);
	ASSERT_MSG(mp_projPsfManager != nullptr,
	           "Error occured during the setup of ProjectionPsfManager");
}

const TimeOfFlightHelper* OperatorProjector::getTOFHelper() const
{
	return mp_tofHelper.get();
}

const ProjectionPsfManager*
    OperatorProjector::getProjectionPsfManager() const
{
	return mp_projPsfManager.get();
}

void OperatorProjector::get_alpha(double r0, double r1, double p1, double p2,
                                    double inv_p12, double& amin, double& amax)
{
	amin = 0.0;
	amax = 1.0;
	if (p1 != p2)
	{
		const double a0 = (r0 - p1) * inv_p12;
		const double a1 = (r1 - p1) * inv_p12;
		if (a0 < a1)
		{
			amin = a0;
			amax = a1;
		}
		else
		{
			amin = a1;
			amax = a0;
		}
	}
	else if (p1 < r0 || p1 > r1)
	{
		amax = 0.0;
		amin = 1.0;
	}
}