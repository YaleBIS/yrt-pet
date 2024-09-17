/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/ListModeLUTDOI.hpp"

#include "datastruct/scanner/Scanner.hpp"
#include "utils/Globals.hpp"

#include <cmath>
#include <cstring>
#include <memory>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_listmodelutdoi(py::module& m)
{
	auto c = py::class_<ListModeLUTDOI, ListModeLUT>(m, "ListModeLUTDOI");

	c.def("writeToFile", &ListModeLUTDOI::writeToFile);

	auto c_alias = py::class_<ListModeLUTDOIAlias, ListModeLUTDOI>(
		m, "ListModeLUTDOIAlias");
	c_alias.def(py::init<Scanner*, bool, int>(), py::arg("scanner"),
	            py::arg("flag_tof") = false, py::arg("numLayers") = 256);

	c_alias.def(
		"bind",
		static_cast<void (ListModeLUTDOIAlias::*)(
			pybind11::array_t<timestamp_t, pybind11::array::c_style>&,
			pybind11::array_t<det_id_t, pybind11::array::c_style>&,
			pybind11::array_t<det_id_t, pybind11::array::c_style>&,
			pybind11::array_t<unsigned char, pybind11::array::c_style>&,
			pybind11::array_t<unsigned char, pybind11::array::c_style>&)>(
			&ListModeLUTDOIAlias::bind),
		py::arg("timestamps"), py::arg("detector_ids1"),
		py::arg("detector_ids2"), py::arg("doi1"), py::arg("doi2"));
	c_alias.def("bind",
	            static_cast<void (ListModeLUTDOIAlias::*)(
		            pybind11::array_t<timestamp_t, pybind11::array::c_style>&,
		            pybind11::array_t<det_id_t, pybind11::array::c_style>&,
		            pybind11::array_t<det_id_t, pybind11::array::c_style>&,
		            pybind11::array_t<unsigned char, pybind11::array::c_style>&,
		            pybind11::array_t<unsigned char, pybind11::array::c_style>&,
		            pybind11::array_t<float, pybind11::array::c_style>&)>(
		            &ListModeLUTDOIAlias::bind),
	            py::arg("timestamps"), py::arg("detector_ids1"),
	            py::arg("detector_ids2"), py::arg("doi1"), py::arg("doi2"),
	            py::arg("tof_ps"));


	auto c_owned = py::class_<ListModeLUTDOIOwned, ListModeLUTDOI>(
		m, "ListModeLUTDOIOwned");
	c_owned.def(py::init<Scanner*, bool, int>(), py::arg("scanner"),
	            py::arg("flag_tof") = false, py::arg("numLayers") = 256);
	c_owned.def(py::init<Scanner*, std::string, bool, int>(),
	            py::arg("scanner"), py::arg("listMode_fname"),
	            py::arg("flag_tof") = false, py::arg("numLayers") = 256);
	c_owned.def("readFromFile", &ListModeLUTDOIOwned::readFromFile);
	c_owned.def("allocate", &ListModeLUTDOIOwned::allocate);
}

#endif  // if BUILD_PYBIND11


ListModeLUTDOI::ListModeLUTDOI(const Scanner* s, bool p_flagTOF,
                               int numLayers)
	: ListModeLUT(s, p_flagTOF),
	  m_numLayers(numLayers) {}

ListModeLUTDOIOwned::ListModeLUTDOIOwned(const Scanner* s, bool p_flagTOF,
                                         int numLayers)
	: ListModeLUTDOI(s, p_flagTOF, numLayers)
{
	mp_timestamps = std::make_unique<Array1D<timestamp_t>>();
	mp_detectorId1 = std::make_unique<Array1D<det_id_t>>();
	mp_detectorId2 = std::make_unique<Array1D<det_id_t>>();
	mp_doi1 = std::make_unique<Array1D<unsigned char>>();
	mp_doi2 = std::make_unique<Array1D<unsigned char>>();
	if (m_flagTOF)
	{
		mp_tof_ps = std::make_unique<Array1D<float>>();
	}
}

ListModeLUTDOIOwned::ListModeLUTDOIOwned(const Scanner* s,
                                         const std::string& listMode_fname,
                                         bool p_flagTOF, int numLayers)
	: ListModeLUTDOIOwned(s, p_flagTOF, numLayers)
{
	readFromFile(listMode_fname);
}

ListModeLUTDOIAlias::ListModeLUTDOIAlias(const Scanner* s, bool p_flagTOF,
                                         int numLayers)
	: ListModeLUTDOI(s, p_flagTOF, numLayers)
{
	mp_timestamps = std::make_unique<Array1DAlias<timestamp_t>>();
	mp_detectorId1 = std::make_unique<Array1DAlias<det_id_t>>();
	mp_detectorId2 = std::make_unique<Array1DAlias<det_id_t>>();
	mp_doi1 = std::make_unique<Array1DAlias<unsigned char>>();
	mp_doi2 = std::make_unique<Array1DAlias<unsigned char>>();
	if (m_flagTOF)
	{
		mp_tof_ps = std::make_unique<Array1DAlias<float>>();
	}
}

void ListModeLUTDOIOwned::readFromFile(const std::string& listMode_fname)
{
	std::ifstream fin(listMode_fname, std::ios::in | std::ios::binary);
	if (!fin.good())
	{
		throw std::runtime_error("Error reading input file " + listMode_fname +
		                         "ListModeLUTDOIOwned::readFromFile.");
	}

	// first check that file has the right size:
	fin.seekg(0, std::ios::end);
	size_t end = fin.tellg();
	fin.seekg(0, std::ios::beg);
	size_t begin = fin.tellg();
	size_t fileSize = end - begin;
	int num_fields = m_flagTOF ? 6 : 5;
	size_t sizeOfAnEvent =
		(num_fields - 2) * sizeof(float) + (2 * sizeof(unsigned char));
	if (fileSize <= 0 || (fileSize % sizeOfAnEvent) != 0)
	{
		throw std::runtime_error("Error: Input file has incorrect size in "
			"ListModeLUTDOIOwned::readFromFile.");
	}

	// Allocate the memory
	size_t numEvents = fileSize / sizeOfAnEvent;
	allocate(numEvents);

	// Read content of file
	size_t numEventsBatch = size_t(1) << 15;
	auto buff =
		std::make_unique<unsigned char[]>(numEventsBatch * sizeOfAnEvent);
	size_t eventStart = 0;
	while (eventStart < numEvents)
	{
		size_t numEventsBatchCurr =
			std::min(numEventsBatch, numEvents - eventStart);
		size_t readSize = numEventsBatchCurr * sizeOfAnEvent;
		fin.read((char*)buff.get(), readSize);

		int num_threads = Globals::get_num_threads();
#pragma omp parallel for num_threads(num_threads)
		for (size_t i = 0; i < numEventsBatchCurr; i++)
		{
			(*mp_timestamps)[eventStart + i] =
				*(reinterpret_cast<float*>(&(buff[sizeOfAnEvent * i])));
			(*mp_detectorId1)[eventStart + i] =
				*(reinterpret_cast<det_id_t*>(&(buff[sizeOfAnEvent * i + 4])));
			(*mp_doi1)[eventStart + i] = buff[sizeOfAnEvent * i + 8];
			(*mp_detectorId2)[eventStart + i] =
				*(reinterpret_cast<det_id_t*>(&(buff[sizeOfAnEvent * i + 9])));
			(*mp_doi2)[eventStart + i] = buff[sizeOfAnEvent * i + 13];
			if (m_flagTOF)
			{
				(*mp_tof_ps)[eventStart + i] = *(
					reinterpret_cast<float*>(&(buff[sizeOfAnEvent * i + 14])));
			}
		}
		eventStart += numEventsBatchCurr;
	}
}

bool ListModeLUTDOI::hasArbitraryLORs() const
{
	return true;
}

line_t ListModeLUTDOI::getArbitraryLOR(bin_t id) const
{
	det_id_t detId1 = getDetector1(id);
	det_id_t detId2 = getDetector2(id);
	Vector3DFloat p1 = mp_scanner->getDetectorPos(detId1);
	Vector3DFloat p2 = mp_scanner->getDetectorPos(detId2);
	Vector3DFloat n1 = mp_scanner->getDetectorOrient(detId1);
	Vector3DFloat n2 = mp_scanner->getDetectorOrient(detId2);
	double layerSize = (1 << 8) / (float)m_numLayers;
	double doi1_t = std::floor((*mp_doi1)[id] / layerSize) *
	                mp_scanner->crystalDepth / (float)m_numLayers;
	double doi2_t = std::floor((*mp_doi2)[id] / layerSize) *
	                mp_scanner->crystalDepth / (float)m_numLayers;
	const Vector3DFloat p1_doi(
		p1.x + (doi1_t - 0.5 * mp_scanner->crystalDepth) * n1.x,
		p1.y + (doi1_t - 0.5 * mp_scanner->crystalDepth) * n1.y,
		p1.z + (doi1_t - 0.5 * mp_scanner->crystalDepth) * n1.z);
	const Vector3DFloat p2_doi(
		p2.x + (doi2_t - 0.5 * mp_scanner->crystalDepth) * n2.x,
		p2.y + (doi2_t - 0.5 * mp_scanner->crystalDepth) * n2.y,
		p2.z + (doi2_t - 0.5 * mp_scanner->crystalDepth) * n2.z);
	return line_t{p1_doi.x, p1_doi.y, p1_doi.z, p2_doi.x, p2_doi.y, p2_doi.z};
}

void ListModeLUTDOI::writeToFile(const std::string& listMode_fname) const
{
	int num_fields = m_flagTOF ? 6 : 5;
	size_t numEvents = count();
	std::ofstream file;
	file.open(listMode_fname.c_str(), std::ios::binary | std::ios::out);
	size_t sizeOfAnEvent =
		(num_fields - 2) * sizeof(float) + (2 * sizeof(unsigned char));

	size_t numEventsBatch = size_t(1) << 15;
	auto buff =
		std::make_unique<unsigned char[]>(numEventsBatch * sizeOfAnEvent);
	size_t eventStart = 0;
	while (eventStart < numEvents)
	{
		size_t numEventsBatchCurr =
			std::min(numEventsBatch, numEvents - eventStart);
		size_t writeSize = numEventsBatchCurr * sizeOfAnEvent;
		for (size_t i = 0; i < numEventsBatchCurr; i++)
		{
			memcpy(&buff[sizeOfAnEvent * i],
			       reinterpret_cast<char*>(&(*mp_timestamps)[eventStart + i]),
			       sizeof(float));
			memcpy(&buff[sizeOfAnEvent * i + 4],
			       reinterpret_cast<char*>(&(*mp_detectorId1)[eventStart + i]),
			       sizeof(det_id_t));
			buff[sizeOfAnEvent * i + 8] = (*mp_doi1)[eventStart + i];
			memcpy(&buff[sizeOfAnEvent * i + 9],
			       reinterpret_cast<char*>(&(*mp_detectorId2)[eventStart + i]),
			       sizeof(det_id_t));
			buff[sizeOfAnEvent * i + 13] = (*mp_doi2)[eventStart + i];
			if (m_flagTOF)
			{
				memcpy(&buff[sizeOfAnEvent * i + 14],
				       reinterpret_cast<char*>(&(*mp_tof_ps)[eventStart + i]),
				       sizeof(float));
			}
		}
		file.write((char*)buff.get(), writeSize);
		eventStart += numEventsBatchCurr;
	}
	file.close();
}

void ListModeLUTDOIOwned::allocate(size_t num_events)
{
	static_cast<Array1D<timestamp_t>*>(mp_timestamps.get())
		->allocate(num_events);
	static_cast<Array1D<det_id_t>*>(mp_detectorId1.get())->allocate(num_events);
	static_cast<Array1D<det_id_t>*>(mp_detectorId2.get())->allocate(num_events);
	static_cast<Array1D<unsigned char>*>(mp_doi1.get())->allocate(num_events);
	static_cast<Array1D<unsigned char>*>(mp_doi2.get())->allocate(num_events);
	if (m_flagTOF)
	{
		static_cast<Array1D<float>*>(mp_tof_ps.get())->allocate(num_events);
	}
}

void ListModeLUTDOIAlias::bind(const Array1DBase<timestamp_t>* pp_timestamps,
                               const Array1DBase<det_id_t>* pp_detector_ids1,
                               const Array1DBase<det_id_t>* pp_detector_ids2,
                               const Array1DBase<unsigned char>* pp_doi1,
                               const Array1DBase<unsigned char>* pp_doi2,
                               const Array1DBase<float>* pp_tof_ps)
{
	static_cast<Array1DAlias<timestamp_t>*>(mp_timestamps.get())
		->bind(*pp_timestamps);
	if (mp_timestamps->getRawPointer() == nullptr)
	{
		throw std::runtime_error("The timestamps array could not be bound");
	}

	static_cast<Array1DAlias<det_id_t>*>(mp_detectorId1.get())
		->bind(*pp_detector_ids1);
	if (mp_detectorId1->getRawPointer() == nullptr)
	{
		throw std::runtime_error("The detector_ids1 array could not be bound");
	}

	static_cast<Array1DAlias<det_id_t>*>(mp_detectorId2.get())
		->bind(*pp_detector_ids2);
	if (mp_detectorId2->getRawPointer() == nullptr)
	{
		throw std::runtime_error("The detector_ids2 array could not be bound");
	}

	static_cast<Array1DAlias<unsigned char>*>(mp_doi1.get())->bind(*pp_doi1);
	if (mp_doi1->getRawPointer() == nullptr)
	{
		throw std::runtime_error("The doi1 array could not be bound");
	}
	static_cast<Array1DAlias<unsigned char>*>(mp_doi2.get())->bind(*pp_doi2);
	if (mp_doi2->getRawPointer() == nullptr)
	{
		throw std::runtime_error("The doi2 array could not be bound");
	}

	if (mp_tof_ps != nullptr && pp_tof_ps != nullptr)
	{
		static_cast<Array1DAlias<float>*>(mp_tof_ps.get())->bind(*pp_tof_ps);
		if (mp_tof_ps->getRawPointer() == nullptr)
			throw std::runtime_error("The tof_ps array could not be bound");
	}
}

#if BUILD_PYBIND11
void ListModeLUTDOIAlias::bind(
	pybind11::array_t<timestamp_t, pybind11::array::c_style>& p_timestamps,
	pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids1,
	pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids2,
	pybind11::array_t<unsigned char, pybind11::array::c_style>& p_doi1,
	pybind11::array_t<unsigned char, pybind11::array::c_style>& p_doi2)
{
	pybind11::buffer_info buffer1 = p_timestamps.request();
	if (buffer1.ndim != 1)
	{
		throw std::invalid_argument(
			"The timestamps buffer has to be 1-dimensional");
	}
	static_cast<Array1DAlias<timestamp_t>*>(mp_timestamps.get())
		->bind(reinterpret_cast<timestamp_t*>(buffer1.ptr), buffer1.shape[0]);

	pybind11::buffer_info buffer2 = p_detector_ids1.request();
	if (buffer2.ndim != 1)
	{
		throw std::invalid_argument(
			"The detector_ids1 buffer has to be 1-dimensional");
	}
	static_cast<Array1DAlias<det_id_t>*>(mp_detectorId1.get())
		->bind(reinterpret_cast<det_id_t*>(buffer2.ptr), buffer2.shape[0]);

	pybind11::buffer_info buffer3 = p_detector_ids2.request();
	if (buffer3.ndim != 1)
	{
		throw std::invalid_argument(
			"The detector_ids2 buffer has to be 1-dimensional");
	}
	static_cast<Array1DAlias<det_id_t>*>(mp_detectorId2.get())
		->bind(reinterpret_cast<det_id_t*>(buffer3.ptr), buffer3.shape[0]);

	pybind11::buffer_info buffer4 = p_doi1.request();
	if (buffer4.ndim != 1)
	{
		throw std::invalid_argument("The doi1 buffer has to be 1-dimensional");
	}
	static_cast<Array1DAlias<unsigned char>*>(mp_doi1.get())
		->bind(reinterpret_cast<unsigned char*>(buffer4.ptr), buffer4.shape[0]);
	pybind11::buffer_info buffer5 = p_doi2.request();
	if (buffer5.ndim != 1)
	{
		throw std::invalid_argument("The doi2 buffer has to be 1-dimensional");
	}
	static_cast<Array1DAlias<unsigned char>*>(mp_doi2.get())
		->bind(reinterpret_cast<unsigned char*>(buffer5.ptr), buffer5.shape[0]);
}

void ListModeLUTDOIAlias::bind(
	pybind11::array_t<timestamp_t, pybind11::array::c_style>& p_timestamps,
	pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids1,
	pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids2,
	pybind11::array_t<unsigned char, pybind11::array::c_style>& p_doi1,
	pybind11::array_t<unsigned char, pybind11::array::c_style>& p_doi2,
	pybind11::array_t<float, pybind11::array::c_style>& p_tof_ps)
{
	bind(p_timestamps, p_detector_ids1, p_detector_ids2, p_doi1, p_doi2);
	if (!m_flagTOF)
		throw std::logic_error(
			"The ListMode was not created with TOF flag at true");
	pybind11::buffer_info buffer = p_tof_ps.request();
	if (buffer.ndim != 1)
	{
		throw std::invalid_argument("The TOF buffer has to be 1-dimensional");
	}
	static_cast<Array1DAlias<float>*>(mp_tof_ps.get())
		->bind(reinterpret_cast<float*>(buffer.ptr), buffer.shape[0]);
}
#endif


std::unique_ptr<ProjectionData>
	ListModeLUTDOIOwned::create(const Scanner& scanner,
	                            const std::string& filename,
	                            const Plugin::OptionsResult& pluginOptions)
{
	bool flagTOF = pluginOptions.find("flag_tof") != pluginOptions.end();

	const auto numLayers_it = pluginOptions.find("num_layers");
	std::unique_ptr<ListModeLUTDOIOwned> lm;
	if (numLayers_it == pluginOptions.end())
	{
		lm = std::make_unique<ListModeLUTDOIOwned>(&scanner, filename,
		                                           flagTOF);
	}
	else
	{
		int numLayers = std::stoi(numLayers_it->second);
		lm = std::make_unique<ListModeLUTDOIOwned>(&scanner, filename,
		                                           flagTOF, numLayers);
	}

	if (pluginOptions.count("lor_motion"))
	{
		lm->addLORMotion(pluginOptions.at("lor_motion"));
	}
	return lm;
}

Plugin::OptionsListPerPlugin ListModeLUTDOIOwned::getOptions()
{
	return {{"flag_tof", {"Flag for reading TOF column", true}},
	        {"num_layers", {"Number of layers", false}},
	        {"lor_motion", {"LOR motion file for motion correction", false}}};
}

REGISTER_PROJDATA_PLUGIN("LM-DOI", ListModeLUTDOIOwned,
                         ListModeLUTDOIOwned::create,
                         ListModeLUTDOIOwned::getOptions)