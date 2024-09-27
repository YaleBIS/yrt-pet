/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/IO.hpp"
#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "geometry/Constants.hpp"
#include "scatter/ScatterEstimator.hpp"
#include "utils/Globals.hpp"
#include "utils/ReconstructionUtils.hpp"

#include <cxxopts.hpp>
#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
	try
	{
		std::string scanner_fname;
		std::string promptsHis_fname;
		std::string normHis_fname;
		std::string sensHis_fname;
		std::string randomsHis_fname;
		std::string imgParams_fname;
		std::string acfHis_fname;
		std::string attImg_fname;
		std::string attImgParams_fname;
		std::string crystalMaterial_name = "LYSO";
		size_t nZ, nPhi, nR;
		std::string outSensImg_fname;
		std::string scatterHistoOut_fname;
		std::string projector_name = "S";
		std::string sourceImage_fname;
		std::string scatterHistoIn_fname;
		int numThreads = -1;
		int num_OSEM_subsets = 1;
		int num_MLEM_iterations = 3;
		bool printProgressFlag = false;
		int maskWidth = -1;
		float acfThreshold = 0.9523809f;  // 1/1.05
		bool saveIntermediary = false;

		// Parse command line arguments
		cxxopts::Options options(argv[0],
		                         "Single-Scatter-Simulation and Scatter "
		                         "Correction histogram generation");
		options.positional_help("[optional args]").show_positional_help();

		/* clang-format off */
		options.add_options()
		("s,scanner", "Scanner parameters file", cxxopts::value(scanner_fname))
		("prompts", "Prompts histogram file", cxxopts::value(promptsHis_fname))
		("norm", "Normalisation histogram file", cxxopts::value(normHis_fname))
		("randoms", "Randoms histogram file", cxxopts::value(randomsHis_fname))
		("sens_his", "Sensitivity histogram file (To use instead of a normalisation histogram)", cxxopts::value(sensHis_fname))
		("acf", "Attenuation coefficients factor", cxxopts::value(acfHis_fname))
		("p,params", "Source image parameters file", cxxopts::value(imgParams_fname))
		("att", "Attenuation image file", cxxopts::value(attImg_fname))
		("att_params", "Attenuation image parameters file", cxxopts::value(attImgParams_fname))
		("crystal_mat", "Crystal material name (default: LYSO)", cxxopts::value(crystalMaterial_name))
		("nZ", "Number of Z planes to consider for SSS", cxxopts::value(nZ))
		("nPhi", "Number of Phi angles to consider for SSS", cxxopts::value(nPhi))
		("nR", "Number of R distances to consider for SSS", cxxopts::value(nR))
		("o,out", "Additive histogram output filename", cxxopts::value(scatterHistoOut_fname))
		("out_sens", "Generated sensitivity image output filename", cxxopts::value(outSensImg_fname))
		("source", "Non scatter-corrected source image (if available)", cxxopts::value(sourceImage_fname))
		("num_threads", "Number of threads", cxxopts::value(numThreads))
		("print_progress", "Print progress flag", cxxopts::value(printProgressFlag))
		("scatter_his", "Previously generated scatter histogram (if available)", cxxopts::value(scatterHistoIn_fname))
		("save_intermediary", "Enable saving intermediary histograms", cxxopts::value(saveIntermediary))
		("mask_width", "Tail fitting mask width. By default, uses 1/10th of the \'r\' dimension", cxxopts::value(maskWidth))
		("acf_threshold", "Tail fitting ACF threshold", cxxopts::value(acfThreshold))
		("num_subsets", "Number of subsets to use for the MLEM part (Default: 1)", cxxopts::value(num_OSEM_subsets))
		("num_iterations", "Number of MLEM iterations to do to generate source image (if needed) (Default: 3)", cxxopts::value(num_MLEM_iterations))
		("projector", "Projector to use, choices: Siddon (S), Distance-Driven (DD)"
			 #if BUILD_CUDA
			 ", or GPU Distance-Driven (DD_GPU)"
			 #endif
			 ". The default projector is Siddon", cxxopts::value<std::string>(projector_name))
		("h,help", "Print help");
		/* clang-format on */

		auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return 0;
		}

		std::vector<std::string> required_params = {
		    "scanner",    "params", "prompts", "acf",  "att",
		    "att_params", "out",    "nZ",      "nPhi", "nR"};
		bool missing_args = false;
		for (auto& p : required_params)
		{
			if (result.count(p) == 0)
			{
				std::cerr << "Argument '" << p << "' missing" << std::endl;
				missing_args = true;
			}
		}
		if (missing_args)
		{
			std::cerr << options.help() << std::endl;
			return -1;
		}

		bool isNorm;
		const std::string* normOrSensHis_fname;
		if (!normHis_fname.empty())
		{
			isNorm = true;
			normOrSensHis_fname = &normHis_fname;
		}
		else if (!sensHis_fname.empty())
		{
			isNorm = false;
			normOrSensHis_fname = &sensHis_fname;
		}
		else
		{
			std::cerr
			    << "You need to provide either a sensitivity histogram or a "
			       "normalisation histogram"
			    << std::endl;
			return -1;
		}

		Globals::set_num_threads(numThreads);
		auto scanner = std::make_unique<Scanner>(scanner_fname);

		// Check if scanner parameters have been set properly for Scatter
		// estimation
		if (scanner->collimatorRadius < 0.0f || scanner->fwhm < 0.0f ||
		    scanner->energyLLD < 0.0f)
		{
			std::cerr
			    << "The scanner parameters given need to have a value for "
			       "\'collimatorRadius\',\'fwhm\', and \'energyLLD\'."
			    << std::endl;
			return -1;
		}

		ImageParams imageParams(imgParams_fname);

		Scatter::CrystalMaterial crystalMaterial =
		    Scatter::getCrystalMaterialFromName(crystalMaterial_name);

		// Attenuation image
		ImageParams attImageParams(attImgParams_fname);
		auto attImg =
		    std::make_unique<ImageOwned>(attImageParams, attImg_fname);

		std::cout << "Reading histograms..." << std::endl;
		auto promptsHis =
		    std::make_unique<Histogram3DOwned>(*scanner, promptsHis_fname);
		auto randomsHis =
		    std::make_unique<Histogram3DOwned>(*scanner, randomsHis_fname);
		auto normOrSensHis =
		    std::make_unique<Histogram3DOwned>(*scanner, *normOrSensHis_fname);
		auto acfHis =
		    std::make_unique<Histogram3DOwned>(*scanner, acfHis_fname);
		std::cout << "Done reading histograms." << std::endl;

		// Generate additive histogram
		std::cout << "Preparing additive histogram..." << std::endl;
		auto additiveHis = std::make_unique<Histogram3DOwned>(*scanner);
		additiveHis->allocate();
		if (isNorm)
		{
			additiveHis->operationOnEachBinParallel(
			    [&randomsHis, &acfHis, &normOrSensHis](bin_t bin) -> float
			    {
				    return randomsHis->getProjectionValue(bin) *
				           normOrSensHis->getProjectionValue(bin) /
				           (acfHis->getProjectionValue(bin) + EPS_FLT);
			    });
		}
		else
		{
			additiveHis->operationOnEachBinParallel(
			    [&randomsHis, &acfHis, &normOrSensHis](bin_t bin) -> float
			    {
				    return randomsHis->getProjectionValue(bin) /
				           (normOrSensHis->getProjectionValue(bin) *
				                acfHis->getProjectionValue(bin) +
				            EPS_FLT);
			    });
		}

		if (saveIntermediary)
		{
			additiveHis->writeToFile(
			    "intermediary_firstAdditiveCorrection.his");
		}

		std::unique_ptr<ImageOwned> sourceImg = nullptr;
		if (sourceImage_fname.empty())
		{
			// Generate histogram to use for the sensitivity image generation
			std::cout << "Preparing sensitivity histogram for the MLEM part..."
			          << std::endl;
			auto sensDataHis = std::make_unique<Histogram3DOwned>(*scanner);
			sensDataHis->allocate();
			if (isNorm)
			{
				sensDataHis->operationOnEachBinParallel(
				    [&acfHis, &normOrSensHis](bin_t bin) -> float
				    {
					    return acfHis->getProjectionValue(bin) /
					           (normOrSensHis->getProjectionValue(bin) +
					            EPS_FLT);
				    });
			}
			else
			{
				sensDataHis->operationOnEachBinParallel(
				    [&acfHis, &normOrSensHis](bin_t bin) -> float
				    {
					    return acfHis->getProjectionValue(bin) *
					           normOrSensHis->getProjectionValue(bin);
				    });
			}

			if (saveIntermediary)
			{
				sensDataHis->writeToFile("intermediary_sensData.his");
			}

			std::vector<std::shared_ptr<Image>> sensImages;
			sourceImg = std::make_unique<ImageOwned>(imageParams);
			sourceImg->allocate();

			// Generate source Image
			auto projectorType = IO::getProjector(projector_name);
			auto osem =
			    Util::createOSEM(*scanner, IO::requiresGPU(projectorType));
			osem->num_MLEM_iterations = num_MLEM_iterations;
			osem->addHis = additiveHis.get();
			osem->imageParams = imageParams;
			osem->outImage = sourceImg.get();
			osem->num_OSEM_subsets = num_OSEM_subsets;
			osem->projectorType = projectorType;
			osem->setSensDataInput(sensDataHis.get());
			osem->setDataInput(promptsHis.get());
			osem->generateSensitivityImages(sensImages, outSensImg_fname);
			osem->registerSensitivityImages(sensImages);
			osem->reconstruct();

			if (saveIntermediary)
			{
				sourceImg->writeToFile("intermediary_sourceImage.img");
			}

			// Deallocate Sensitivity data histogram to save memory
			sensDataHis = nullptr;
		}
		else
		{
			if (!outSensImg_fname.empty())
			{
				std::cerr
				    << "Warning: The sensitivity image will not be generated "
				       "since a source image was already provided"
				    << std::endl;
			}
			sourceImg =
			    std::make_unique<ImageOwned>(imageParams, sourceImage_fname);
		}

		Scatter::ScatterEstimator sss(*scanner, *sourceImg, *attImg,
		                              promptsHis.get(), normOrSensHis.get(),
		                              randomsHis.get(), acfHis.get(),
		                              crystalMaterial, 13, isNorm, maskWidth,
		                              acfThreshold, saveIntermediary);

		if (!scatterHistoIn_fname.empty())
		{
			auto scatterHis = std::make_shared<Histogram3DOwned>(
			    *scanner, scatterHistoIn_fname);
			sss.setScatterHistogram(scatterHis);
		}

		sss.computeAdditiveScatterCorrection(nZ, nPhi, nR, printProgressFlag);

		std::cout << "Preparing final additive correction histogram..."
		          << std::endl;
		const Histogram3D* scatterHis = sss.getScatterHistogram();
		additiveHis->operationOnEachBinParallel(
		    [&additiveHis, &scatterHis](const bin_t bin) -> float
		    {
			    return additiveHis->getProjectionValue(bin) +
			           scatterHis->getProjectionValue(bin);
		    });

		std::cout << "Saving histogram file..." << std::endl;
		additiveHis->writeToFile(scatterHistoOut_fname);

		std::cout << "Done." << std::endl;
		return 0;
	}
	catch (const cxxopts::exceptions::exception& e)
	{
		std::cerr << "Error parsing options: " << e.what() << std::endl;
		return -1;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Caught exception: " << e.what() << std::endl;
		return -1;
	}
}
