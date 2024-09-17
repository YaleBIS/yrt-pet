/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/Image.hpp"
#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "geometry/Cylinder.hpp"
#include "geometry/Plane.hpp"

#include "omp.h"

class GCSingleScatterSimulation
{
public:
	GCSingleScatterSimulation(Scanner* p_scanner, Image* p_lambda,
	                          Image* p_mu, Histogram3D* p_prompts_histo,
	                          Histogram3D* p_norm_histo,
	                          Histogram3D* p_acf_histo,
	                          const std::string& mu_det_file, int seedi = 13,
	                          bool p_doTailFitting = true);

	void readMuDetFile(const std::string& mu_det_file);
	void run_SSS(size_t numberZ, size_t numberPhi, size_t numberR,
	             bool printProgress = false);
	double compute_single_scatter_in_lor(StraightLineParam* lor);

	Histogram3DOwned* getScatterHistogram() { return mp_scatterHisto.get(); }

protected:
	double ran1(int* idum);
	double get_mu_scaling_factor(double energy);
	double get_klein_nishina(double cosa);
	double get_intersection_length_lor_crystal(StraightLineParam* lor);
	bool pass_collimator(StraightLineParam* lor);
	double get_mu_det(double energy);

public:
	int nsamples;
	std::vector<double> xsamp, ysamp, zsamp;                // mu image samples
	std::vector<size_t> samples_z, samples_phi, samples_r;  // Histogram samples
	float energy_lld, sigma_energy;
	float rdet, thickdet, afovdet, rcoll;

protected:
	Scanner* mp_scanner;
	Histogram3D* mp_promptsHisto;
	Histogram3D* mp_normHisto;
	Histogram3D* mp_acfHisto;
	Image* mp_mu;      // Attenuation image
	Image* mp_lambda;  // Image from one iteration
	Cylinder m_cyl1, m_cyl2;
	Plane m_endPlate1, m_endPlate2;

	std::unique_ptr<double[]> mp_muDetTable;
	bool m_doTailFitting;
	const float m_maskThreshold = 1.05;

	std::unique_ptr<Histogram3DOwned> mp_scatterHisto;  // Final structure
};