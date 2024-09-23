/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/image/ImageDevice.cuh"

#include "datastruct/image/Image.hpp"
#include "datastruct/image/ImageSpaceKernels.cuh"
#include "operators/OperatorDevice.cuh"
#include "utils/Assert.hpp"
#include "utils/GPUMemory.cuh"
#include "utils/GPUTypes.cuh"
#include "utils/GPUUtils.cuh"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

void py_setup_imagedevice(py::module& m)
{
	auto c = py::class_<ImageDevice, ImageBase>(m, "ImageDevice");
	c.def(
	    "transferToDeviceMemory",
	    [](ImageDevice& self, py::buffer& np_data)
	    {
		    try
		    {
			    py::buffer_info buffer = np_data.request();
			    if (buffer.ndim != 3)
			    {
				    throw std::invalid_argument(
				        "The buffer given has to have 3 dimensions");
			    }
			    if (buffer.format != py::format_descriptor<float>::format())
			    {
				    throw std::invalid_argument(
				        "The buffer given has to have a float32 format");
			    }
			    const std::vector dims = {self.getParams().nz,
			                              self.getParams().ny,
			                              self.getParams().nx};
			    for (int i = 0; i < 3; i++)
			    {
				    if (buffer.shape[i] != dims[i])
				    {
					    throw std::invalid_argument(
					        "The buffer shape does not match with the image "
					        "parameters");
				    }
			    }
			    self.transferToDeviceMemory(
			        reinterpret_cast<float*>(buffer.ptr), true);
		    }
		    catch (const std::exception& e)
		    {
			    std::cerr << "Error in given buffer: " << e.what() << std::endl;
		    }
	    },
	    "Copy from numpy array to device", "numpy_array"_a);
	c.def(
	    "transferToDeviceMemory",
	    [](ImageDevice& self, const Image* sourceImage)
	    { self.transferToDeviceMemory(sourceImage, true); },
	    "Copy from a Host-side Image object", "sourceImage"_a);
	c.def(
	    "transferToHostMemory",
	    [](const ImageDevice& self)
	    {
		    const auto& params = self.getParams();
		    const size_t elemStride = sizeof(float);

		    const size_t size = params.nz * params.ny * params.nx;
		    float* float_ptr = new float[size];
		    self.transferToHostMemory(float_ptr, true);

		    // Create a Python object that will free the allocated
		    // memory when destroyed:
		    py::capsule free_when_done(
		        float_ptr,
		        [](void* f)
		        {
			        auto* float_ptr = reinterpret_cast<float*>(f);
			        std::cerr << "Element [0] = " << float_ptr[0] << "\n";
			        std::cerr << "freeing memory @ " << f << "\n";
			        delete[] float_ptr;
		        });

		    return py::array_t<float, py::array::c_style>(
		        {params.nz, params.ny, params.nx},  // dimensions
		        {elemStride * params.nx * params.ny, elemStride * params.nx,
		         elemStride},     // strides
		        float_ptr,        // the data pointer
		        free_when_done);  // deleter
	    },
	    "Transfer device image to host-side memory and return it as a numpy "
	    "array");
	c.def(
	    "transferToHostMemory", [](ImageDevice& self, Image* img)
	    { self.transferToHostMemory(img); },
	    "Transfer device image to host-side Image", "image"_a);

	c.def("setValue", &ImageDevice::setValue, "initValue"_a = 0.0);
	c.def("addFirstImageToSecond", &ImageDevice::addFirstImageToSecond,
	      "imgOut"_a);
	c.def("applyThreshold", &ImageDevice::applyThreshold, "maskImage"_a,
	      "threshold"_a, "val_le_scale"_a, "val_le_off"_a, "val_gt_scale"_a,
	      "val_gt_off"_a);
	c.def("updateEMThreshold", &ImageDevice::updateEMThreshold, "updateImg"_a,
	      "normImg"_a, "threshold"_a);
	c.def("writeToFile", &ImageDevice::writeToFile, "image_fname"_a);

	c.def("applyThresholdDevice", &ImageDevice::applyThresholdDevice,
	      "maskImg"_a, "threshold"_a, "val_le_scale"_a, "val_le_off"_a,
	      "val_gt_scale"_a, "val_gt_off"_a);

	auto c_owned =
	    py::class_<ImageDeviceOwned, ImageDevice>(m, "ImageDeviceOwned");
	c_owned.def(
	    py::init(
	        [](const ImageParams& imgParams) {
		        return std::make_unique<ImageDeviceOwned>(imgParams, nullptr);
	        }),
	    "Create ImageDevice using image parameters (will not allocate)",
	    "img_params"_a);
	c_owned.def(
	    py::init(
	        [](const ImageParams& imgParams, const std::string& filename) {
		        return std::make_unique<ImageDeviceOwned>(imgParams, filename,
		                                                    nullptr);
	        }),
	    "Create ImageDevice using image parameters and filename",
	    "img_params"_a, "filename"_a);
	c_owned.def(
	    py::init(
	        [](const Image* img_ptr)
	        { return std::make_unique<ImageDeviceOwned>(img_ptr, nullptr); }),
	    "Create a ImageDevice using a host-size Image", "img"_a);
	c_owned.def("allocate",
	            [](ImageDeviceOwned& self) { self.allocate(true); });

	auto c_alias =
	    py::class_<ImageDeviceAlias, ImageDevice>(m, "ImageDeviceAlias");
	c_alias.def(
	    py::init(
	        [](const ImageParams& imgParams) {
		        return std::make_unique<ImageDeviceAlias>(imgParams, nullptr);
	        }),
	    "Create ImageDevice using image parameters (will not allocate)",
	    "img_params"_a);
	c_alias.def("getDevicePointer", &ImageDeviceAlias::getDevicePointerInULL);
	c_alias.def("setDevicePointer",
	            static_cast<void (ImageDeviceAlias::*)(size_t)>(
	                &ImageDeviceAlias::setDevicePointer),
	            "Set a device address for the image array. For "
	            "usage with PyTorch, use \'myArray.data_ptr()\'",
	            "data_ptr"_a);
	c_alias.def("isDevicePointerSet", &ImageDeviceAlias::isDevicePointerSet,
	            "Returns true if the device pointer is not null");
}

#endif  // if BUILD_PYBIND11


ImageDevice::ImageDevice(const ImageParams& imgParams,
                             const cudaStream_t* stream_ptr)
    : ImageBase(imgParams), mp_stream(stream_ptr)
{
	m_launchParams = Util::initiateDeviceParameters(imgParams);
	m_imgSize = imgParams.nx * imgParams.ny * imgParams.nz;
}

const cudaStream_t* ImageDevice::getStream() const
{
	return mp_stream;
}

size_t ImageDevice::getImageSize() const
{
	return m_imgSize;
}

void ImageDevice::transferToDeviceMemory(const float* hp_img_ptr,
                                           bool p_synchronize)
{
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");
	Util::copyHostToDevice(getDevicePointer(), hp_img_ptr, m_imgSize, mp_stream,
	                       p_synchronize);
}

void ImageDevice::transferToDeviceMemory(const Image* hp_img_ptr,
                                           bool p_synchronize)
{
	ASSERT_MSG(getParams().isSameDimensionsAs(hp_img_ptr->getParams()),
	           "Image dimensions mismatch");

	m_tempBuffer.reAllocateIfNeeded(m_imgSize);
	float* h_floatTempBuffer_ptr = m_tempBuffer.getPointer();
	const double* h_double_ptr = hp_img_ptr->getData().getRawPointer();
	ASSERT_MSG(h_double_ptr != nullptr,
	           "Either the image given is invalid or not allocated");

	for (int id = 0; id < static_cast<int>(m_imgSize); id++)
	{
		h_floatTempBuffer_ptr[id] = static_cast<float>(h_double_ptr[id]);
	}
	transferToDeviceMemory(h_floatTempBuffer_ptr, p_synchronize);
}

void ImageDevice::transferToHostMemory(float* hp_img_ptr,
                                         bool p_synchronize) const
{
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");
	Util::copyDeviceToHost(hp_img_ptr, getDevicePointer(), m_imgSize, mp_stream,
	                       p_synchronize);
}

void ImageDevice::transferToHostMemory(Image* hp_img_ptr,
                                         bool p_synchronize) const
{
	m_tempBuffer.reAllocateIfNeeded(m_imgSize);
	float* tempBufPointer = m_tempBuffer.getPointer();
	std::cout << "Transferring image from Device to Host..." << std::endl;
	transferToHostMemory(tempBufPointer, p_synchronize);
	std::cout << "Done transferring image from Device to Host." << std::endl;

	ASSERT(hp_img_ptr != nullptr);
	// Note: Eventually we'll need to use float32 for images in order to avoid
	// caveats like this
	double* hp_double_ptr = hp_img_ptr->getData().getRawPointer();
	ASSERT_MSG(hp_double_ptr != nullptr,
	           "The Image provided is not yet allocated or bound");
	for (int id = 0; id < static_cast<int>(m_imgSize); id++)
	{
		hp_double_ptr[id] = static_cast<double>(tempBufPointer[id]);
	}
}

void ImageDevice::applyThresholdDevice(const ImageDevice* maskImg,
                                         const float threshold,
                                         const float val_le_scale,
                                         const float val_le_off,
                                         const float val_gt_scale,
                                         const float val_gt_off)
{
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");

	if (mp_stream != nullptr)
	{
		applyThreshold_kernel<<<m_launchParams.gridSize,
		                        m_launchParams.blockSize, 0, *mp_stream>>>(
		    getDevicePointer(), maskImg->getDevicePointer(), threshold,
		    val_le_scale, val_le_off, val_gt_scale, val_gt_off, getParams().nx,
		    getParams().ny, getParams().nz);
		cudaStreamSynchronize(*mp_stream);
	}
	else
	{
		applyThreshold_kernel<<<m_launchParams.gridSize,
		                        m_launchParams.blockSize>>>(
		    getDevicePointer(), maskImg->getDevicePointer(), threshold,
		    val_le_scale, val_le_off, val_gt_scale, val_gt_off, getParams().nx,
		    getParams().ny, getParams().nz);
		cudaDeviceSynchronize();
	}
	cudaCheckError();
}

void ImageDevice::applyThreshold(const ImageBase* maskImg, double threshold,
                                   double val_le_scale, double val_le_off,
                                   double val_gt_scale, double val_gt_off)
{
	const auto maskImg_ImageDevice =
	    dynamic_cast<const ImageDevice*>(maskImg);
	ASSERT_MSG(maskImg_ImageDevice != nullptr,
	           "Input image has the wrong type");

	applyThresholdDevice(
	    maskImg_ImageDevice, static_cast<float>(threshold),
	    static_cast<float>(val_le_scale), static_cast<float>(val_le_off),
	    static_cast<float>(val_gt_scale), static_cast<float>(val_gt_off));
}

void ImageDevice::writeToFile(const std::string& image_fname) const
{
	auto tmpImage = std::make_unique<ImageOwned>(getParams());
	tmpImage->allocate();
	transferToHostMemory(tmpImage.get(), true);
	tmpImage->writeToFile(image_fname);
}

void ImageDevice::updateEMThreshold(ImageBase* updateImg,
                                      const ImageBase* normImg,
                                      double threshold)
{
	auto* updateImg_ImageDevice = dynamic_cast<ImageDevice*>(updateImg);
	const auto* normImg_ImageDevice =
	    dynamic_cast<const ImageDevice*>(normImg);

	ASSERT_MSG(updateImg_ImageDevice != nullptr,
	           "updateImg is not ImageDevice");
	ASSERT_MSG(normImg != nullptr, "normImg is not ImageDevice");
	ASSERT_MSG(
	    updateImg_ImageDevice->getParams().isSameDimensionsAs(getParams()),
	    "Image dimensions mismatch");
	ASSERT_MSG(
	    normImg_ImageDevice->getParams().isSameDimensionsAs(getParams()),
	    "Image dimensions mismatch");
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");

	if (mp_stream != nullptr)
	{
		updateEM_kernel<<<m_launchParams.gridSize, m_launchParams.blockSize, 0,
		                  *mp_stream>>>(
		    updateImg_ImageDevice->getDevicePointer(), getDevicePointer(),
		    normImg_ImageDevice->getDevicePointer(), getParams().nx,
		    getParams().ny, getParams().nz, threshold);
		cudaStreamSynchronize(*mp_stream);
	}
	else
	{
		updateEM_kernel<<<m_launchParams.gridSize, m_launchParams.blockSize>>>(
		    updateImg_ImageDevice->getDevicePointer(), getDevicePointer(),
		    normImg_ImageDevice->getDevicePointer(), getParams().nx,
		    getParams().ny, getParams().nz, threshold);
		cudaDeviceSynchronize();
	}
	cudaCheckError();
}

void ImageDevice::addFirstImageToSecond(ImageBase* second) const
{
	auto* second_ImageDevice = dynamic_cast<ImageDevice*>(second);

	ASSERT_MSG(second_ImageDevice != nullptr, "imgOut is not ImageDevice");
	ASSERT_MSG(
	    second_ImageDevice->getParams().isSameDimensionsAs(getParams()),
	    "Image dimensions mismatch");
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");

	if (mp_stream != nullptr)
	{
		addFirstImageToSecond_kernel<<<
		    m_launchParams.gridSize, m_launchParams.blockSize, 0, *mp_stream>>>(
		    getDevicePointer(), second_ImageDevice->getDevicePointer(),
		    getParams().nx, getParams().ny, getParams().nz);
		cudaStreamSynchronize(*mp_stream);
	}
	else
	{
		addFirstImageToSecond_kernel<<<m_launchParams.gridSize,
		                               m_launchParams.blockSize>>>(
		    getDevicePointer(), second_ImageDevice->getDevicePointer(),
		    getParams().nx, getParams().ny, getParams().nz);
		cudaDeviceSynchronize();
	}
	cudaCheckError();
}

void ImageDevice::setValue(double initValue)
{
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");

	if (mp_stream != nullptr)
	{
		setValue_kernel<<<m_launchParams.gridSize, m_launchParams.blockSize, 0,
		                  *mp_stream>>>(
		    getDevicePointer(), static_cast<float>(initValue), getParams().nx,
		    getParams().ny, getParams().nz);
		cudaStreamSynchronize(*mp_stream);
	}
	else
	{
		setValue_kernel<<<m_launchParams.gridSize, m_launchParams.blockSize>>>(
		    getDevicePointer(), static_cast<float>(initValue), getParams().nx,
		    getParams().ny, getParams().nz);
		cudaDeviceSynchronize();
	}
	cudaCheckError();
}


ImageDeviceOwned::ImageDeviceOwned(const ImageParams& imgParams,
                                       const cudaStream_t* stream_ptr)
    : ImageDevice(imgParams, stream_ptr), mpd_devicePointer(nullptr)
{
}

ImageDeviceOwned::ImageDeviceOwned(const ImageParams& imgParams,
                                       const std::string& filename,
                                       const cudaStream_t* stream_ptr)
    : ImageDevice(imgParams, stream_ptr), mpd_devicePointer(nullptr)
{
	readFromFile(filename);
}

ImageDeviceOwned::ImageDeviceOwned(const Image* img_ptr,
                                       const cudaStream_t* stream_ptr)
    : ImageDevice(img_ptr->getParams(), stream_ptr),
      mpd_devicePointer(nullptr)
{
	allocate(false);
	transferToDeviceMemory(img_ptr);
}

ImageDeviceOwned::~ImageDeviceOwned()
{
	if (mpd_devicePointer != nullptr)
	{
		std::cout << "Freeing image device buffer..." << std::endl;
		Util::deallocateDevice(mpd_devicePointer);
	}
}

void ImageDeviceOwned::allocate(bool synchronize)
{
	const auto& params = getParams();
	std::cout << "Allocating device memory for an image of dimensions "
	          << "[" << params.nz << ", " << params.ny << ", " << params.nx
	          << "]..." << std::endl;

	Util::allocateDevice(&mpd_devicePointer, m_imgSize, mp_stream, false);
	Util::memsetDevice(mpd_devicePointer, 0, m_imgSize, mp_stream, synchronize);

	std::cout << "Done allocating device memory." << std::endl;
}

void ImageDeviceOwned::readFromFile(const std::string& filename)
{
	// Create temporary Image
	const auto img = std::make_unique<ImageOwned>(getParams(), filename);
	allocate(false);
	transferToDeviceMemory(img.get(), true);
}

float* ImageDeviceOwned::getDevicePointer()
{
	return mpd_devicePointer;
}

const float* ImageDeviceOwned::getDevicePointer() const
{
	return mpd_devicePointer;
}


ImageDeviceAlias::ImageDeviceAlias(const ImageParams& imgParams,
                                       const cudaStream_t* stream_ptr)
    : ImageDevice(imgParams, stream_ptr), mpd_devicePointer(nullptr)
{
}

float* ImageDeviceAlias::getDevicePointer()
{
	return mpd_devicePointer;
}

const float* ImageDeviceAlias::getDevicePointer() const
{
	return mpd_devicePointer;
}

size_t ImageDeviceAlias::getDevicePointerInULL() const
{
	return reinterpret_cast<size_t>(mpd_devicePointer);
}

void ImageDeviceAlias::setDevicePointer(float* ppd_devicePointer)
{
	mpd_devicePointer = ppd_devicePointer;
}

void ImageDeviceAlias::setDevicePointer(size_t ppd_pointerInULL)
{
	setDevicePointer(reinterpret_cast<float*>(ppd_pointerInULL));
}

bool ImageDeviceAlias::isDevicePointerSet() const
{
	return mpd_devicePointer != nullptr;
}
