/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

/* **************************************************************************************
 * Def.: Post-reconstruction correction of motion by warping images obtained by
 *       motion-divided independant reconstruction of a PET acquisition to a
 *frame of
 *       reference.
 * TODO:
 *		- Add verbose to the process.
 *		- Add frame specifier in parser and adapt the code accordingly.
 *		- Add frame disabling in parser and set it in the warper.
 * *************************************************************************************/

#include "datastruct/image/GCImage.hpp"
#include "motion/ImageWarperFunction.hpp"

#include <cxxopts.hpp>

#include <fstream>
#include <string>
#include <vector>


int main(int argc, char* argv[])
{
	// Input variables without default.
	std::vector<std::string> im_fname;
	std::string warpParamFile;
	std::string outFile;
	std::string outParamFile;

	// Parse command line arguments
	try
	{
		cxxopts::Options options(argv[0],
		                         "Post-reconstruction motion correction "
		                         "driver");
		options.positional_help("[optional args]").show_positional_help();

		options.add_options()("i,im", "Paths to each images", cxxopts::value(im_fname))
		("p,param", "Image parameters file", cxxopts::value<std::string>(outParamFile))
		("w,wFile", "Path to the warp parameters file", cxxopts::value(warpParamFile))
		("o,out", "Where the resulting image will be saved", cxxopts::value(outFile))
		("help", "Print help");

		auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return -1;
		}

		std::vector<std::string> required_params = {"im", "wFile", "out",
		                                            "param"};
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
	}
	catch (const cxxopts::exceptions::exception& e)
	{
		std::cerr << "Error parsing options: " << e.what() << std::endl;
		exit(1);
	}

	// Load the images.
	GCImageParams imgParams(outParamFile);
	std::vector<std::unique_ptr<GCImageOwned>> imageList(im_fname.size());
	for (size_t i = 0; i < im_fname.size(); i++)
	{
		std::ifstream f(im_fname[i].c_str());
		if (f.good())
		{
			imageList[i] = std::make_unique<GCImageOwned>(imgParams);
			imageList[i]->allocate();
			imageList[i]->readFromFile(im_fname[i]);
		}
		else
		{
			std::cerr << "The file " << im_fname[i] << " does not exist."
			          << std::endl;
			return -1;
		}
	}

	ImageWarperFunction warper;
	warper.setImageHyperParam(imgParams);
	warper.setFramesParamFromFile(warpParamFile);

	auto postMotionCorrImage = std::make_unique<GCImageOwned>(imgParams);
	postMotionCorrImage->allocate();
	for (size_t m = 0; m < imageList.size(); m++)
	{
		warper.warpImageToRefFrame(imageList[m].get(), m);
		imageList[m]->addFirstImageToSecond(postMotionCorrImage.get());
	}

	postMotionCorrImage->writeToFile(outFile);

	return EXIT_SUCCESS;
}
