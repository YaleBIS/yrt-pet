/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/projection/Histogram3D.hpp"
#include "scatter/Crystal.hpp"
#include "scatter/SingleScatterSimulator.hpp"

class Scanner;
class Image;

namespace Scatter
{
	class ScatterEstimator
	{
	public:
		ScatterEstimator(const Scanner& pr_scanner,
		                   const Image& pr_lambda, const Image& pr_mu,
		                   const Histogram3D* pp_promptsHis,
		                   const Histogram3D* pp_normOrSensHis,
		                   const Histogram3D* pp_randomsHis,
		                   const Histogram3D* pp_acfHis,
		                   CrystalMaterial p_crystalMaterial, int seedi,
		                   bool p_doTailFitting, bool isNorm, int maskWidth,
		                   float maskThreshold, bool saveIntermediary);

		void estimateScatter(size_t numberZ, size_t numberPhi, size_t numberR,
		                     bool printProgress = false);

		const Histogram3DOwned* getScatterHistogram() const;

	protected:
		static void generateScatterTailsMask(const Histogram3D& acfHis,
		                                     std::vector<bool>& mask,
		                                     size_t maskWidth,
		                                     float maskThreshold);

		void saveScatterTailsMask();

	private:
		const Scanner& mr_scanner;
		SingleScatterSimulator m_sss;
		const Histogram3D* mp_promptsHis;
		const Histogram3D* mp_randomsHis;
		const Histogram3D* mp_normOrSensHis;
		const Histogram3D* mp_acfHis;

		std::vector<bool> m_scatterTailsMask;
		bool m_doTailFitting;
		bool m_isNorm;
		bool m_saveIntermediary;
		float m_maskThreshold;
		size_t m_scatterTailsMaskWidth;

		std::unique_ptr<Histogram3DOwned> mp_scatterHisto;  // Final structure
	};
}  // namespace Scatter
