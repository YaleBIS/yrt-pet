/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../GCPluginOptionsHelper.hpp"
#include "datastruct/GCIO.hpp"
#include "datastruct/projection/GCHistogram3D.hpp"
#include "datastruct/projection/GCSparseHistogram.hpp"
#include "datastruct/scanner/GCScanner.hpp"
#include "utils/GCGlobals.hpp"
#include "utils/GCReconstructionUtils.hpp"

#include <cxxopts.hpp>
#include <iostream>


int main(int argc, char** argv)
{
	std::string scanner_fname;
	std::string input_fname;
	std::string input_format;
	std::string out_fname;
	bool toSparseHistogram = false;
	int numThreads = -1;

	Plugin::OptionsResult pluginOptionsResults;  // For plugins' options

	// Parse command line arguments
	try
	{
		cxxopts::Options options(
		    argv[0], "Convert any input format to a histogram");
		options.positional_help("[optional args]").show_positional_help();
		/* clang-format off */
		options.add_options()
		("s,scanner", "Scanner parameters file name", cxxopts::value<std::string>(scanner_fname))
		("i,input", "Input file", cxxopts::value<std::string>(input_fname))
		("f,format", "Input file format. Possible values: " + IO::possibleFormats(), cxxopts::value<std::string>(input_format))
		("o,out", "Output histogram filename", cxxopts::value<std::string>(out_fname))
		("sparse", "Convert to a sparse histogram instead", cxxopts::value<bool>(toSparseHistogram))
		("num_threads", "Number of threads to use", cxxopts::value<int>(numThreads))
		("h,help", "Print help");
		/* clang-format on */

		// Add plugin options
		PluginOptionsHelper::fillOptionsFromPlugins(options);

		auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return -1;
		}

		std::vector<std::string> required_params = {"scanner", "input", "out",
		                                            "format"};
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

		// Parse plugin options
		pluginOptionsResults =
		    PluginOptionsHelper::convertPluginResultsToMap(result);
	}
	catch (const cxxopts::exceptions::exception& e)
	{
		std::cerr << "Error parsing options: " << e.what() << std::endl;
		return -1;
	}

	GCGlobals::set_num_threads(numThreads);

	auto scanner = std::make_unique<GCScannerOwned>(scanner_fname);

	std::cout << "Reading input data..." << std::endl;

	std::unique_ptr<IProjectionData> dataInput = IO::openProjectionData(
	    input_fname, input_format, *scanner, pluginOptionsResults);

	std::cout << "Done reading input data." << std::endl;

	if (toSparseHistogram)
	{
		std::cout << "Accumulating into sparse histogram..." << std::endl;
		auto sparseHisto =
		    std::make_unique<GCSparseHistogram>(*scanner, *dataInput);
		std::cout << "Saving sparse histogram..." << std::endl;
		sparseHisto->writeToFile(out_fname);
	}
	else
	{
		std::cout << "Preparing output Histogram3D..." << std::endl;
		auto histoOut = std::make_unique<GCHistogram3DOwned>(scanner.get());
		histoOut->allocate();
		histoOut->clearProjections(0.0f);
		std::cout << "Done preparing output Histogram3D." << std::endl;

		std::cout << "Accumulating into Histogram3D..." << std::endl;
		if (IO::isFormatListMode(input_format))
		{
			// ListMode input, use atomic to accumulate
			Util::convertToHistogram3D<true>(*dataInput, *histoOut);
		}
		else
		{
			// Histogram input, no need to use atomic to accumulate
			Util::convertToHistogram3D<false>(*dataInput, *histoOut);
		}

		std::cout << "Histogram3D generated.\nWriting file..." << std::endl;
		histoOut->writeToFile(out_fname);
	}

	std::cout << "Done." << std::endl;

	return 0;
}
