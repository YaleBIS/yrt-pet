/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../PluginOptionsHelper.hpp"
#include "datastruct/IO.hpp"
#include "datastruct/projection/GCHistogram3D.hpp"
#include "datastruct/projection/GCSparseHistogram.hpp"
#include "datastruct/scanner/GCScanner.hpp"
#include "utils/GCAssert.hpp"
#include "utils/GCGlobals.hpp"
#include "utils/GCReconstructionUtils.hpp"

#include <cxxopts.hpp>
#include <iostream>


int main(int argc, char** argv)
{
	std::string scanner_fname;
	std::vector<std::string> input_fnames;
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
		("i,input", "Input histogram files (separated by commas)", cxxopts::value<std::vector<std::string>>(input_fnames))
		("f,format", "Input files format. Possible values: " + IO::possibleFormats(Plugin::InputFormatsChoice::ONLYHISTOGRAMS), cxxopts::value<std::string>(input_format))
		("o,out", "Output histogram filename", cxxopts::value<std::string>(out_fname))
		("sparse", "Convert to a sparse histogram instead", cxxopts::value<bool>(toSparseHistogram))
		("num_threads", "Number of threads to use", cxxopts::value<int>(numThreads))
		("h,help", "Print help");
		/* clang-format on */

		// Add plugin options
		PluginOptionsHelper::fillOptionsFromPlugins(
		    options, Plugin::InputFormatsChoice::ONLYHISTOGRAMS);

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

	std::unique_ptr<IHistogram> histoOut;
	if (toSparseHistogram)
	{
		histoOut = std::make_unique<GCSparseHistogram>(*scanner);
	}
	else
	{
		auto histo3DOut = std::make_unique<GCHistogram3DOwned>(scanner.get());
		histo3DOut->allocate();
		histo3DOut->clearProjections(0.0f);
		histoOut = std::move(histo3DOut);
	}

	bool histo3DToHisto3D = input_format == "H";

	for (const auto& input_fname : input_fnames)
	{
		std::cout << "Reading input data..." << std::endl;

		std::unique_ptr<IProjectionData> dataInput = IO::openProjectionData(
		    input_fname, input_format, *scanner, pluginOptionsResults);

		std::cout << "Done reading input data." << std::endl;

		if (toSparseHistogram)
		{
			auto* sparseHisto =
			    reinterpret_cast<GCSparseHistogram*>(histoOut.get());
			std::cout << "Accumulating into sparse histogram..." << std::endl;
			sparseHisto->allocate(sparseHisto->count() + dataInput->count());
			sparseHisto->accumulate<true>(*dataInput);
		}
		else
		{
			auto histo3DOut = reinterpret_cast<GCHistogram3D*>(histoOut.get());
			if (histo3DToHisto3D)
			{
				std::cout << "Adding Histogram3Ds..." << std::endl;
				const auto* dataInputHisto3D =
				    dynamic_cast<const GCHistogram3D*>(dataInput.get());
				ASSERT(dataInputHisto3D != nullptr);

				histo3DOut->getData() += dataInputHisto3D->getData();
			}
			else
			{
				std::cout << "Accumulating Histogram into Histogram3D..."
				          << std::endl;
				Util::convertToHistogram3D<false>(*dataInput, *histo3DOut);
			}
		}
	}

	if (toSparseHistogram)
	{
		const auto* sparseHisto =
		    reinterpret_cast<const GCSparseHistogram*>(histoOut.get());
		std::cout << "Saving output sparse histogram..." << std::endl;
		sparseHisto->writeToFile(out_fname);
	}
	else
	{
		const auto* histo3DOut =
		    reinterpret_cast<const GCHistogram3D*>(histoOut.get());
		std::cout << "Saving output Histogram3D..." << std::endl;
		histo3DOut->writeToFile(out_fname);
	}

	std::cout << "Done." << std::endl;

	return 0;
}
