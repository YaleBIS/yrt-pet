/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/scanner/Scanner.hpp"

#include "datastruct/scanner/DetRegular.hpp"
#include "utils/JSONUtils.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>

namespace py = pybind11;
using namespace py::literals;

void py_setup_scanner(pybind11::module& m)
{
	auto c = py::class_<Scanner>(m, "Scanner");

	c.def(py::init<std::string, float, float, float, float, float, size_t,
	               size_t, size_t, size_t, size_t, size_t>(),
	      "scannerName"_a, "axialFOV"_a, "crystalSize_z"_a,
	      "crystalSize_trans"_a, "crystalDepth"_a, "scannerRadius"_a,
	      "detsPerRing"_a, "numRings"_a, "numDOI"_a, "maxRingDiff"_a,
	      "minAngDiff"_a, "detsPerBlock"_a);
	c.def(py::init<const std::string&>());
	c.def("getNumDets", &Scanner::getNumDets);
	c.def("getTheoreticalNumDets", &Scanner::getTheoreticalNumDets);
	c.def("getDetectorPos", &Scanner::getDetectorPos);
	c.def("getDetectorOrient", &Scanner::getDetectorOrient);
	c.def_readwrite("scannerName", &Scanner::scannerName);
	c.def_readwrite("axialFOV", &Scanner::axialFOV);
	c.def_readwrite("crystalSize_z", &Scanner::crystalSize_z);
	c.def_readwrite("crystalSize_trans", &Scanner::crystalSize_trans);
	c.def_readwrite("crystalDepth", &Scanner::crystalDepth);
	c.def_readwrite("scannerRadius", &Scanner::scannerRadius);
	c.def_readwrite("collimatorRadius", &Scanner::collimatorRadius);
	c.def_readwrite("fwhm", &Scanner::fwhm);
	c.def_readwrite("energyLLD", &Scanner::energyLLD);
	c.def_readwrite("dets_per_ring", &Scanner::dets_per_ring);
	c.def_readwrite("fwhm", &Scanner::fwhm);
	c.def_readwrite("num_rings", &Scanner::num_rings);
	c.def_readwrite("num_doi", &Scanner::num_doi);
	c.def_readwrite("max_ring_diff", &Scanner::max_ring_diff);
	c.def_readwrite("min_ang_diff", &Scanner::min_ang_diff);
	c.def_readwrite("dets_per_block", &Scanner::dets_per_block);
	c.def("createLUT",
	      [](const Scanner& s)
	      {
		      auto lut = std::make_unique<Array2D<float>>();
		      s.createLUT(*lut.get());
		      size_t shape[2]{s.getNumDets(), 6};
		      auto lut_array = py::array_t<float>(
		          shape, std::move(lut.get())->getRawPointer());
		      return lut_array;
	      });
	c.def("readFromFile", &Scanner::readFromFile);
	c.def("readFromString", &Scanner::readFromString);
	c.def(pybind11::pickle([](const Scanner& s) { return s.getScannerPath(); },
	                       [](const std::string& fname)
	                       {
		                       std::unique_ptr<Scanner> s =
		                           std::make_unique<Scanner>(fname);
		                       return s;
	                       }));
}
#endif

Scanner::Scanner(std::string pr_scannerName, float p_axialFOV,
                 float p_crystalSize_z, float p_crystalSize_trans,
                 float p_crystalDepth, float p_scannerRadius,
                 size_t p_detsPerRing, size_t p_numRings, size_t p_numDOI,
                 size_t p_maxRingDiff, size_t p_minAngDiff,
                 size_t p_detsPerBlock)
    : scannerName(std::move(pr_scannerName)),
      axialFOV(p_axialFOV),
      crystalSize_z(p_crystalSize_z),
      crystalSize_trans(p_crystalSize_trans),
      crystalDepth(p_crystalDepth),
      scannerRadius(p_scannerRadius),
      collimatorRadius(p_scannerRadius - p_crystalDepth),
      fwhm(0.2),
      energyLLD(400),
      dets_per_ring(p_detsPerRing),
      num_rings(p_numRings),
      num_doi(p_numDOI),
      max_ring_diff(p_maxRingDiff),
      min_ang_diff(p_minAngDiff),
      dets_per_block(p_detsPerBlock)
{
}

Scanner::Scanner(const std::string& p_definitionFile)
{
	readFromFile(p_definitionFile);
}

size_t Scanner::getNumDets() const
{
	return mp_detectors->getNumDets();
}

size_t Scanner::getTheoreticalNumDets() const
{
	return num_doi * num_rings * dets_per_ring;
}

Vector3DFloat Scanner::getDetectorPos(det_id_t id) const
{
	return mp_detectors->getPos(id);
}

Vector3DFloat Scanner::getDetectorOrient(det_id_t id) const
{
	return mp_detectors->getOrient(id);
}

std::shared_ptr<DetectorSetup> Scanner::getDetectorSetup() const
{
	return mp_detectors;
}

bool Scanner::isValid() const
{
	return mp_detectors != nullptr;
}

void Scanner::createLUT(Array2D<float>& lut) const
{
	lut.allocate(this->getNumDets(), 6);
	for (size_t i = 0; i < this->getNumDets(); i++)
	{
		const Vector3DFloat pos = mp_detectors->getPos(i);
		const Vector3DFloat orient = mp_detectors->getOrient(i);
		lut[i][0] = pos.x;
		lut[i][1] = pos.y;
		lut[i][2] = pos.z;
		lut[i][3] = orient.x;
		lut[i][4] = orient.y;
		lut[i][5] = orient.z;
	}
}

void Scanner::setDetectorSetup(
    const std::shared_ptr<DetectorSetup>& pp_detectors)
{
	mp_detectors = pp_detectors;
}

void Scanner::readFromString(const std::string& fileContents)
{
	json j;
	try
	{
		j = json::parse(fileContents);
	}
	catch (json::exception& e)
	{
		throw std::invalid_argument("Error in Scanner JSON file parsing");
	}

	float scannerFileVersion = 0.0;
	Util::getParam<float>(&j, &scannerFileVersion, "VERSION", 0.0, true,
	                      "Missing VERSION in scanner definition file");

	Util::getParam<std::string>(
	    &j, &scannerName, "scannerName", "", true,
	    "Missing scanner name in scanner definition file");

	std::string detCoord;
	const bool isDetCoordGiven =
	    Util::getParam<std::string>(&j, &detCoord, "detCoord", "", false);

	Util::getParam<float>(
	    &j, &axialFOV, "axialFOV", 0.0, true,
	    "Missing Axial Field Of View value in scanner definition file");
	Util::getParam<float>(
	    &j, &crystalSize_trans, "crystalSize_trans", 0.0, true,
	    "Missing transaxial crystal size in scanner definition file");
	Util::getParam<float>(
	    &j, &crystalSize_z, "crystalSize_z", 0.0, true,
	    "Missing z-axis crystal size in scanner definition file");

	// Optional values for Scatter estimation only
	Util::getParam<float>(&j, &collimatorRadius, "collimatorRadius", -1.0,
	                      false);
	Util::getParam<float>(&j, &fwhm, "fwhm", -1.0, false);
	Util::getParam<float>(&j, &energyLLD, "energyLLD", -1.0, false);

	Util::getParam<float>(&j, &crystalDepth, "crystalDepth", 0.0, true,
	                      "Missing crystal depth in scanner definition file");
	Util::getParam<float>(&j, &scannerRadius, "scannerRadius", 0.0, true,
	                      "Missing scanner radius in scanner definition file");
	Util::getParam<size_t>(&j, &dets_per_ring, "dets_per_ring", 0, true);
	Util::getParam<size_t>(&j, &num_rings, "num_rings", 0, true);
	Util::getParam<size_t>(&j, &num_doi, "num_doi", 0, true);
	Util::getParam<size_t>(&j, &max_ring_diff, "max_ring_diff", 0, true);
	Util::getParam<size_t>(&j, &min_ang_diff, "min_ang_diff", 0, true);
	Util::getParam<size_t>(&j, &dets_per_block, "dets_per_block", 1, false);

	// Check for errors
	if (scannerFileVersion != SCANNER_FILE_VERSION)
	{
		throw std::invalid_argument(
		    "Wrong file version for Scanner JSON file, the "
		    "current version is " +
		    std::to_string(SCANNER_FILE_VERSION));
	}

	// Join paths for DetCoord
	if (isDetCoordGiven)
	{
		const fs::path detCoord_path =
		    m_scannerPath.parent_path() / fs::path(detCoord);
		mp_detectors = std::make_shared<DetCoordOwned>(detCoord_path.string());
		if (mp_detectors->getNumDets() != getTheoreticalNumDets())
			throw std::runtime_error(
			    "The number of detectors given by the LUT file does not match "
			    "the scanner's characteristics. Namely, (num_doi * num_rings * "
			    "dets_per_ring) does not equal the size of the LUT");
	}
	else
	{
		mp_detectors = std::make_shared<DetRegular>(this);
		reinterpret_cast<DetRegular*>(mp_detectors.get())->generateLUT();
	}
}

std::string Scanner::getScannerPath() const
{
	return m_scannerPath.string();
}

void Scanner::readFromFile(const std::string& p_definitionFile)
{
	m_scannerPath = fs::path(p_definitionFile);
	if (!(fs::exists(m_scannerPath)))
	{
		throw std::invalid_argument(
		    "The scanner definition file given does not exist");
	}

	std::ifstream i(m_scannerPath.string());
	auto ss = std::ostringstream{};
	ss << i.rdbuf();
	std::string fileContents = ss.str();
	readFromString(fileContents);
}
