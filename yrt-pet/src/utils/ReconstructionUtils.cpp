/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/ReconstructionUtils.hpp"

#include "datastruct/IO.hpp"
#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/projection/ListModeLUT.hpp"
#include "geometry/Matrix.hpp"
#include "operators/OperatorProjectorDD.hpp"
#include "operators/OperatorProjectorSiddon.hpp"
#include "recon/OSEM_CPU.hpp"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"
#include "utils/ProgressDisplayMultiThread.hpp"
#include "utils/Tools.hpp"

#if BUILD_CUDA
#include "operators/OperatorProjectorDD_GPU.cuh"
#include "recon/OSEM_GPU.cuh"
#endif


#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;

void py_setup_reconstructionutils(pybind11::module& m)
{
	m.def("histogram3DToListModeLUT", &Util::histogram3DToListModeLUT);
	m.def(
	    "convertToHistogram3D", [](const Histogram& dat, Histogram3D& histoOut)
	    { Util::convertToHistogram3D<false>(dat, histoOut); },
	    py::arg("histogramDataInput"), py::arg("histoOut"));
	m.def(
	    "convertToHistogram3D", [](const ListMode& dat, Histogram3D& histoOut)
	    { Util::convertToHistogram3D<true>(dat, histoOut); },
	    py::arg("listmodeDataInput"), py::arg("histoOut"));
	m.def(
	    "createOSEM",
	    [](const Scanner& scanner, bool useGPU)
	    {
		    auto osem = Util::createOSEM(scanner, useGPU);
		    osem->enableNeedToMakeCopyOfSensImage();
		    return osem;
	    },
	    py::arg("scanner"), py::arg("useGPU") = false);
	m.def("generateTORRandomDOI", &Util::generateTORRandomDOI,
	      py::arg("scanner"), py::arg("d1"), py::arg("d2"), py::arg("vmax"));

	m.def("forwProject",
	      static_cast<void (*)(const Scanner& scanner, const Image& img,
	                           ProjectionData& projData,
	                           OperatorProjector::ProjectorType projectorType)>(
	          &Util::forwProject),
	      py::arg("scanner"), py::arg("img"), py::arg("projData"),
	      py::arg("projectorType") = OperatorProjector::ProjectorType::SIDDON);
	m.def("forwProject",
	      static_cast<void (*)(const Scanner& scanner, const Image& img,
	                           ProjectionData& projData,
	                           const BinIterator& binIterator,
	                           OperatorProjector::ProjectorType projectorType)>(
	          &Util::forwProject),
	      py::arg("scanner"), py::arg("img"), py::arg("projData"),
	      py::arg("binIterator"),
	      py::arg("projectorType") = OperatorProjector::SIDDON);
	m.def("forwProject",
	      static_cast<void (*)(const Image& img, ProjectionData& projData,
	                           const OperatorProjectorParams& projParams,
	                           OperatorProjector::ProjectorType projectorType)>(
	          &Util::forwProject),
	      py::arg("img"), py::arg("projData"), py::arg("projParams"),
	      py::arg("projectorType") = OperatorProjector::SIDDON);

	m.def("backProject",
	      static_cast<void (*)(const Scanner& scanner, Image& img,
	                           const ProjectionData& projData,
	                           OperatorProjector::ProjectorType projectorType)>(
	          &Util::backProject),
	      py::arg("scanner"), py::arg("img"), py::arg("projData"),
	      py::arg("projectorType") = OperatorProjector::ProjectorType::SIDDON);
	m.def("backProject",
	      static_cast<void (*)(const Scanner& scanner, Image& img,
	                           const ProjectionData& projData,
	                           const BinIterator& binIterator,
	                           OperatorProjector::ProjectorType projectorType)>(
	          &Util::backProject),
	      py::arg("scanner"), py::arg("img"), py::arg("projData"),
	      py::arg("binIterator"),
	      py::arg("projectorType") = OperatorProjector::SIDDON);
	m.def("backProject",
	      static_cast<void (*)(Image& img, const ProjectionData& projData,
	                           const OperatorProjectorParams& projParams,
	                           OperatorProjector::ProjectorType projectorType)>(
	          &Util::backProject),
	      py::arg("img"), py::arg("projData"), py::arg("projParams"),
	      py::arg("projectorType") = OperatorProjector::SIDDON);
}

#endif

namespace Util
{
	void histogram3DToListModeLUT(const Histogram3D* histo,
	                              ListModeLUTOwned* lmOut, size_t numEvents)
	{
		ASSERT(lmOut != nullptr);
		const float* dataPtr = histo->getData().getRawPointer();

		// Phase 1: calculate sum of histogram values
		double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
		for (bin_t binId = 0; binId < histo->count(); binId++)
		{
			sum += dataPtr[binId];
		}

		// Default target number of events (histogram sum)
		if (numEvents == 0)
		{
			numEvents = std::lround(sum);
		}
		// Phase 2: calculate actual number of events
		size_t sumInt = 0.0;
#pragma omp parallel for reduction(+ : sumInt)
		for (bin_t binId = 0; binId < histo->count(); binId++)
		{
			sumInt += std::lround(dataPtr[binId] / sum * (double)numEvents);
		}

		// Allocate list-mode data
		lmOut->allocate(sumInt);

		int numThreads = Globals::get_num_threads();
		if (numThreads > 1)
		{
			size_t numBinsPerThread =
			    std::ceil(double(histo->count()) / (double)numThreads);
			Array1D<size_t> partialSums;
			partialSums.allocate(numThreads);

			// Phase 3: prepare partial sums for parallelization
#pragma omp parallel for num_threads(numThreads)
			for (int ti = 0; ti < numThreads; ti++)
			{
				bin_t binStart = ti * numBinsPerThread;
				bin_t binEnd = std::min(histo->count() - 1,
				                        binStart + numBinsPerThread - 1);
				for (bin_t binId = binStart; binId <= binEnd; binId++)
				{
					partialSums[ti] +=
					    std::lround(dataPtr[binId] / sum * (double)numEvents);
				}
			}

			// Calculate indices
			Array1D<size_t> lmStartIdx;
			lmStartIdx.allocate(numThreads);
			lmStartIdx[0] = 0;
			for (int ti = 1; ti < numThreads; ti++)
			{
				lmStartIdx[ti] = lmStartIdx[ti - 1] + partialSums[ti - 1];
			}

			// Phase 4: create events by block
#pragma omp parallel for num_threads(numThreads)
			for (int ti = 0; ti < numThreads; ti++)
			{
				bin_t binStart = ti * numBinsPerThread;
				bin_t binEnd = std::min(histo->count() - 1,
				                        binStart + numBinsPerThread - 1);
				bin_t eventId = lmStartIdx[ti];
				for (bin_t binId = binStart; binId <= binEnd; binId++)
				{
					if (dataPtr[binId] != 0.f)
					{
						auto [d1, d2] = histo->getDetectorPair(binId);
						int numEventsBin = std::lround(dataPtr[binId] / sum *
						                               (double)numEvents);
						for (int ei = 0; ei < numEventsBin; ei++)
						{
							lmOut->setDetectorIdsOfEvent(eventId++, d1, d2);
						}
					}
				}
			}
		}
		else
		{
			bin_t eventId = 0;
			for (bin_t binId = 0; binId < histo->count(); binId++)
			{
				if (dataPtr[binId] != 0.f)
				{
					auto [d1, d2] = histo->getDetectorPair(binId);
					int numEventsBin =
					    std::lround(dataPtr[binId] / sum * (double)numEvents);
					for (int ei = 0; ei < numEventsBin; ei++)
					{
						lmOut->setDetectorIdsOfEvent(eventId++, d1, d2);
					}
				}
			}
		}
	}

	template <bool RequiresAtomicAccumulation>
	void convertToHistogram3D(const ProjectionData& dat, Histogram3D& histoOut)
	{
		float* histoDataPointer = histoOut.getData().getRawPointer();
		const size_t numDatBins = dat.count();

		ProgressDisplayMultiThread progressBar(Globals::get_num_threads(),
		                                       numDatBins, 5);
		progressBar.start();

		const Histogram3D* histoOut_constptr = &histoOut;
		const ProjectionData* dat_constptr = &dat;
#pragma omp parallel for default(none)                            \
    firstprivate(histoDataPointer, numDatBins, histoOut_constptr, \
                     dat_constptr) shared(progressBar)
		for (bin_t datBin = 0; datBin < numDatBins; ++datBin)
		{
			progressBar.progress(omp_get_thread_num(), 1);

			const float projValue = dat_constptr->getProjectionValue(datBin);
			if (projValue > 0)
			{
				const auto [d1, d2] = dat_constptr->getDetectorPair(datBin);
				if (d1 == d2)
				{
					// Do not crash
					continue;
				}
				const bin_t histoBin =
				    histoOut_constptr->getBinIdFromDetPair(d1, d2);
				if constexpr (RequiresAtomicAccumulation)
				{
#pragma omp atomic
					histoDataPointer[histoBin] += projValue;
				}
				else
				{
					histoDataPointer[histoBin] += projValue;
				}
			}
		}
		progressBar.finish();
	}

	template void convertToHistogram3D<true>(const ProjectionData&,
	                                         Histogram3D&);
	template void convertToHistogram3D<false>(const ProjectionData&,
	                                          Histogram3D&);

	Line3D getNativeLOR(const Scanner& scanner, const ProjectionData& dat,
	                    bin_t binId)
	{
		const auto [d1, d2] = dat.getDetectorPair(binId);
		const Vector3D p1 = scanner.getDetectorPos(d1);
		const Vector3D p2 = scanner.getDetectorPos(d2);
		return Line3D{p1, p2};
	}

	void convertProjectionValuesToACF(ProjectionData& dat, float unitFactor)
	{
		dat.operationOnEachBinParallel(
			[&dat, unitFactor](bin_t bin) -> float
			{
				return Util::getAttenuationCoefficientFactor(
					dat.getProjectionValue(bin), unitFactor);
			});
	}

	std::tuple<Line3D, Vector3D, Vector3D>
	    generateTORRandomDOI(const Scanner& scanner, det_id_t d1, det_id_t d2,
	                         int vmax)
	{
		const Vector3D p1 = scanner.getDetectorPos(d1);
		const Vector3D p2 = scanner.getDetectorPos(d2);
		const Vector3D n1 = scanner.getDetectorOrient(d1);
		const Vector3D n2 = scanner.getDetectorOrient(d2);
		const float doi_1_t = (rand() % vmax) /
		                      (static_cast<float>(1 << 8) - 1) *
		                      scanner.crystalDepth;
		const float doi_2_t = (rand() % vmax) /
		                      (static_cast<float>(1 << 8) - 1) *
		                      scanner.crystalDepth;
		const Vector3D p1_doi{p1.x + doi_1_t * n1.x, p1.y + doi_1_t * n1.y,
		                      p1.z + doi_1_t * n1.z};
		const Vector3D p2_doi{p2.x + doi_2_t * n2.x, p2.y + doi_2_t * n2.y,
		                      p2.z + doi_2_t * n2.z};
		const Line3D lorDOI{p1_doi, p2_doi};
		return {lorDOI, n1, n2};
	}

	std::unique_ptr<OSEM> createOSEM(const Scanner& scanner, bool useGPU)
	{
		std::unique_ptr<OSEM> osem;
		if (useGPU)
		{
#if BUILD_CUDA
			osem = std::make_unique<OSEM_GPU>(scanner);
#else
			throw std::runtime_error(
			    "Project not compiled with CUDA, GPU projectors unavailable.");
#endif
		}
		else
		{
			osem = std::make_unique<OSEM_CPU>(scanner);
		}
		return osem;
	}


	// Forward and backward projections
	template <bool IS_FWD>
	static void project(Image* img, ProjectionData* projData,
	                    const OperatorProjectorParams& projParams,
	                    OperatorProjector::ProjectorType projectorType)
	{
		std::unique_ptr<OperatorProjectorBase> oper;
		if (projectorType == OperatorProjector::SIDDON)
		{
			oper = std::make_unique<OperatorProjectorSiddon>(projParams);
		}
		else if (projectorType == OperatorProjector::DD)
		{
			oper = std::make_unique<OperatorProjectorDD>(projParams);
		}
		else if (projectorType == OperatorProjector::DD_GPU)
		{
#ifdef BUILD_CUDA
			oper = std::make_unique<OperatorProjectorDD_GPU>(projParams);
#else
			throw std::runtime_error("GPU projector not supported because "
			                         "Project was not compiled with CUDA ");
#endif
		}
		else
		{
			throw std::logic_error(
			    "Error in forwProject: Unknown projector type (Note that the "
			    "GPU Distance-Driven projector is unsupported for now)");
		}

		if constexpr (IS_FWD)
		{
			std::cout << "Forward projecting all LORs ..." << std::endl;
			oper->applyA(img, projData);
			std::cout << "Done forward projecting all LORs." << std::endl;
		}
		else
		{
			std::cout << "Back projecting all LORs ..." << std::endl;
			oper->applyAH(projData, img);
			std::cout << "Done back projecting all LORs." << std::endl;
		}
	}

	void forwProject(const Scanner& scanner, const Image& img,
	                 ProjectionData& projData,
	                 OperatorProjector::ProjectorType projectorType)
	{
		const auto binIter = projData.getBinIter(1, 0);
		const OperatorProjectorParams projParams(binIter.get(), scanner);
		forwProject(img, projData, projParams, projectorType);
	}

	void forwProject(const Scanner& scanner, const Image& img,
	                 ProjectionData& projData, const BinIterator& binIterator,
	                 OperatorProjector::ProjectorType projectorType)
	{
		const OperatorProjectorParams projParams(&binIterator, scanner);
		forwProject(img, projData, projParams, projectorType);
	}

	void forwProject(const Image& img, ProjectionData& projData,
	                 const OperatorProjectorParams& projParams,
	                 OperatorProjector::ProjectorType projectorType)
	{
		project<true>(const_cast<Image*>(&img), &projData, projParams,
		              projectorType);
	}

	void backProject(const Scanner& scanner, Image& img,
	                 const ProjectionData& projData,
	                 OperatorProjector::ProjectorType projectorType)
	{
		const auto binIter = projData.getBinIter(1, 0);
		const OperatorProjectorParams projParams(binIter.get(), scanner);
		backProject(img, projData, projParams, projectorType);
	}

	void backProject(const Scanner& scanner, Image& img,
	                 const ProjectionData& projData,
	                 const BinIterator& binIterator,
	                 OperatorProjector::ProjectorType projectorType)
	{
		const OperatorProjectorParams projParams(&binIterator, scanner);
		backProject(img, projData, projParams, projectorType);
	}

	void backProject(Image& img, const ProjectionData& projData,
	                 const OperatorProjectorParams& projParams,
	                 OperatorProjector::ProjectorType projectorType)
	{
		project<false>(&img, const_cast<ProjectionData*>(&projData), projParams,
		               projectorType);
	}

}  // namespace Util