/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "scatter/ScatterEstimator.hpp"

#include "datastruct/image/Image.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "geometry/Constants.hpp"
#include "scatter/Crystal.hpp"
#include "utils/ReconstructionUtils.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals;

void py_setup_scatterestimator(py::module& m)
{
	auto c = py::class_<Scatter::ScatterEstimator>(m, "ScatterEstimator");
	c.def(
		py::init<const Scanner&, const Image&, const Image&,
		         const Histogram3D*, const Histogram3D*,
		         const Histogram3D*, const Histogram3D*,
		         Scatter::CrystalMaterial, int, bool, bool, int, float, bool>(),
		"scanner"_a, "source_image"_a, "attenuation_image"_a, "prompts_his"_a,
		"norm_or_sens_his"_a, "randoms_his"_a, "acf_his"_a,
		"crystal_material"_a, "seed"_a, "do_tail_fitting"_a, "is_norm"_a,
		"mask_width"_a, "mask_threshold"_a, "save_intermediary"_a);
	c.def("estimateScatter", &Scatter::ScatterEstimator::estimateScatter,
	      "num_z"_a, "num_phi"_a, "num_r"_a, "print_progress"_a = false);
	c.def("getScatterHistogram",
	      &Scatter::ScatterEstimator::getScatterHistogram);
}
#endif

namespace Scatter
{
	ScatterEstimator::ScatterEstimator(
		const Scanner& pr_scanner, const Image& pr_lambda,
		const Image& pr_mu, const Histogram3D* pp_promptsHis,
		const Histogram3D* pp_normOrSensHis,
		const Histogram3D* pp_randomsHis, const Histogram3D* pp_acfHis,
		CrystalMaterial p_crystalMaterial, int seedi, bool p_doTailFitting,
		bool isNorm, int maskWidth, float maskThreshold, bool saveIntermediary)
		: mr_scanner(pr_scanner),
		  m_sss(pr_scanner, pr_mu, pr_lambda, p_crystalMaterial, seedi)
	{
		mp_promptsHis = pp_promptsHis;
		mp_normOrSensHis = pp_normOrSensHis;
		mp_randomsHis = pp_randomsHis;
		mp_acfHis = pp_acfHis;
		m_doTailFitting = p_doTailFitting;
		m_isNorm = isNorm;
		if (maskWidth > 0)
		{
			m_scatterTailsMaskWidth = maskWidth;
		}
		else
		{
			m_scatterTailsMaskWidth = mp_promptsHis->n_r / 10;
		}
		m_maskThreshold = maskThreshold;
		m_saveIntermediary = saveIntermediary;

		// Initialize buffers
		mp_scatterHisto = std::make_unique<Histogram3DOwned>(mr_scanner);
		mp_scatterHisto->allocate();
		mp_scatterHisto->clearProjections();
	}

	void ScatterEstimator::estimateScatter(size_t numberZ, size_t numberPhi,
	                                       size_t numberR, bool printProgress)
	{
		m_sss.runSSS(numberZ, numberPhi, numberR, *mp_scatterHisto,
		             printProgress);

		if (m_doTailFitting)
		{
			std::cout << "Generating scatter tails mask..." << std::endl;
			generateScatterTailsMask(*mp_acfHis, m_scatterTailsMask,
			                         m_scatterTailsMaskWidth, m_maskThreshold);
			if (m_saveIntermediary)
			{
				saveScatterTailsMask();
			}

			std::cout << "Computing Tail-fit factor..." << std::endl;
			float scat = 0.0f, prompt = 0.0f;
			for (bin_t bin = 0; bin < mp_scatterHisto->count(); bin++)
			{
				// Only fit outside the image
				if (!m_scatterTailsMask[bin])
					continue;

				scat += mp_scatterHisto->getProjectionValue(bin);
				if (m_isNorm)
				{
					prompt += (mp_promptsHis->getProjectionValue(bin) -
					           mp_randomsHis->getProjectionValue(bin)) *
						mp_normOrSensHis->getProjectionValue(bin);
				}
				else
				{
					prompt +=
						(mp_promptsHis->getProjectionValue(bin) -
						 mp_randomsHis->getProjectionValue(bin)) /
						(mp_normOrSensHis->getProjectionValue(bin) + EPS_FLT);
				}
			}
			const float fac = prompt / scat;
			std::cout << "Tail-fitting factor: " << fac << std::endl;
			mp_scatterHisto->getData() *= fac;
		}

		if (m_saveIntermediary)
		{
			mp_scatterHisto->writeToFile("intermediary_scatterEstimate.his");
		}

		std::cout << "Dividing by the ACF..." << std::endl;
		mp_scatterHisto->operationOnEachBin(
			[this](bin_t bin) -> float
			{
				const float acf = mp_acfHis->getProjectionValue(bin);
				if (acf > 0.0f)
				{
					return mp_scatterHisto->getProjectionValue(bin) / acf;
				}
				return 0.0f;
			});
		std::cout << "Done with scatter estimate." << std::endl;
	}


	const Histogram3DOwned* ScatterEstimator::getScatterHistogram() const
	{
		return mp_scatterHisto.get();
	}

	void ScatterEstimator::saveScatterTailsMask()
	{
		const auto tmpHisto = std::make_unique<Histogram3DOwned>(mr_scanner);
		tmpHisto->allocate();
		tmpHisto->operationOnEachBinParallel(
			[this](bin_t bin) -> float
			{
				return m_scatterTailsMask[bin] ? 1.0 : 0.0;
			});
		tmpHisto->writeToFile("intermediary_scatterTailsMask.his");
	}


	void ScatterEstimator::generateScatterTailsMask(
		const Histogram3D& acfHis, std::vector<bool>& mask, size_t maskWidth,
		float maskThreshold)
	{
		const size_t numBins = acfHis.count();
		mask.resize(numBins);
		std::fill(mask.begin(), mask.end(), false);

		for (bin_t binId = 0; binId < numBins; binId++)
		{
			const float acfValue = acfHis.getProjectionValue(binId);
			mask[binId] = acfValue == 0.0 /* For invalid acf bins */ ||
			              acfValue > maskThreshold;
		}

		for (size_t zBin = 0; zBin < acfHis.n_z_bin; zBin++)
		{
			for (size_t phi = 0; phi < acfHis.n_phi; phi++)
			{
				const size_t initRowBinId =
					acfHis.getBinIdFromCoords(0, phi, zBin);

				// Process beginning of the mask
				for (size_t r = 0; r < acfHis.n_r; r++)
				{
					const bin_t binId = initRowBinId + r;
					if (mask[binId] == false)
					{
						if (r > maskWidth)
						{
							// Put zeros from the beginning of the row to the
							// current position minus the width of the mask
							for (bin_t newBinId = initRowBinId;
							     newBinId < binId - maskWidth; newBinId++)
							{
								mask[newBinId] = false;
							}
						}
						break;
					}
				}

				// Process end of the mask
				for (long r = acfHis.n_r - 1; r >= 0; r--)
				{
					const bin_t binId = initRowBinId + r;
					if (mask[binId] == false)
					{
						if (r < static_cast<long>(acfHis.n_r - maskWidth))
						{
							// Put zeros from the beginning of the row to the
							// current position minus the width of the mask
							for (long newR = acfHis.n_r - 1;
							     newR >=
							     static_cast<long>(r + maskWidth);
							     newR--)
							{
								mask[newR + initRowBinId] = false;
							}
						}
						break;
					}
				}
			}
		}
	}

} // namespace Scatter