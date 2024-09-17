/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/GCOperatorProjectorSiddon.hpp"

#include "datastruct/image/Image.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "utils/GCAssert.hpp"
#include "utils/GCGlobals.hpp"
#include "utils/GCReconstructionUtils.hpp"

#include <algorithm>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_gcoperatorprojectorsiddon(py::module& m)
{
	auto c = py::class_<GCOperatorProjectorSiddon, GCOperatorProjector>(
	    m, "GCOperatorProjectorSiddon");
	c.def(py::init<const GCOperatorProjectorParams&>(), py::arg("projParams"));
	c.def_property("num_rays", &GCOperatorProjectorSiddon::getNumRays,
	               &GCOperatorProjectorSiddon::setNumRays);
	c.def(
	    "forward_projection",
	    [](const GCOperatorProjectorSiddon& self, const Image* in_image,
	       const StraightLineParam& lor, const Vector3D& n1,
	       const Vector3D& n2, const GCTimeOfFlightHelper* tofHelper,
	       float tofValue) -> double {
		    return self.forwardProjection(in_image, lor, n1, n2, tofHelper,
		                                  tofValue);
	    },
	    py::arg("in_image"), py::arg("lor"), py::arg("n1"), py::arg("n2"),
	    py::arg("tofHelper") = nullptr, py::arg("tofValue") = 0.0f);
	c.def(
	    "back_projection",
	    [](const GCOperatorProjectorSiddon& self, Image* in_image,
	       const StraightLineParam& lor, const Vector3D& n1,
	       const Vector3D& n2, double proj_value,
	       const GCTimeOfFlightHelper* tofHelper, float tofValue) -> void
	    {
		    self.backProjection(in_image, lor, n1, n2, proj_value, tofHelper,
		                        tofValue);
	    },
	    py::arg("in_image"), py::arg("lor"), py::arg("n1"), py::arg("n2"),
	    py::arg("proj_value"), py::arg("tofHelper") = nullptr,
	    py::arg("tofValue") = 0.0f);
	c.def_static(
	    "single_back_projection",
	    [](Image* in_image, const StraightLineParam& lor, double proj_value,
	       const GCTimeOfFlightHelper* tofHelper, float tofValue) -> void
	    {
		    GCOperatorProjectorSiddon::singleBackProjection(
		        in_image, lor, proj_value, tofHelper, tofValue);
	    },
	    py::arg("in_image"), py::arg("lor"), py::arg("proj_value"),
	    py::arg("tofHelper") = nullptr, py::arg("tofValue") = 0.0f);
	c.def_static(
	    "single_forward_projection",
	    [](const Image* in_image, const StraightLineParam& lor,
	       const GCTimeOfFlightHelper* tofHelper, float tofValue) -> double
	    {
		    return GCOperatorProjectorSiddon::singleForwardProjection(
		        in_image, lor, tofHelper, tofValue);
	    },
	    py::arg("in_image"), py::arg("lor"), py::arg("tofHelper") = nullptr,
	    py::arg("tofValue") = 0.0f);
}
#endif

GCOperatorProjectorSiddon::GCOperatorProjectorSiddon(
    const GCOperatorProjectorParams& p_projParams)
    : GCOperatorProjector(p_projParams), m_numRays(p_projParams.numRays)
{
	if (m_numRays > 1)
	{
		mp_lineGen = std::make_unique<std::vector<MultiRayGenerator>>(
		    GCGlobals::get_num_threads(),
		    MultiRayGenerator(scanner->crystalSize_z,
		                        scanner->crystalSize_trans));
	}
}

int GCOperatorProjectorSiddon::getNumRays() const
{
	return m_numRays;
}

void GCOperatorProjectorSiddon::setNumRays(int n)
{
	m_numRays = n;
}

double GCOperatorProjectorSiddon::forwardProjection(const Image* img,
                                                    const ProjectionData* dat,
                                                    bin_t bin)
{
	auto [lor, tofValue, randomsEstimate, n1, n2] =
	    Util::getProjectionProperties(*scanner, *dat, bin);

	// TODO: What to do with randomsEstimate ?

	return forwardProjection(img, lor, n1, n2, mp_tofHelper.get(), tofValue);
}

void GCOperatorProjectorSiddon::backProjection(Image* img,
                                               const ProjectionData* dat,
                                               bin_t bin, double projValue)
{
	auto [lor, tofValue, randomsEstimate, n1, n2] =
	    Util::getProjectionProperties(*scanner, *dat, bin);

	backProjection(img, lor, n1, n2, projValue, mp_tofHelper.get(), tofValue);
}

double GCOperatorProjectorSiddon::forwardProjection(
    const Image* img, const StraightLineParam& lor, const Vector3D& n1,
    const Vector3D& n2, const GCTimeOfFlightHelper* tofHelper,
    float tofValue) const
{
	const ImageParams& params = img->getParams();
	const Vector3D offsetVec = {params.off_x, params.off_y, params.off_z};

	double imProj = 0.;

	// Avoid multi-ray siddon on attenuation image
	const int numRaysToCast =
	    (img == attImage || img == attImageForBackprojection) ? 1 : m_numRays;

	int currThread = 0;
	if (numRaysToCast > 1)
	{
		currThread = omp_get_thread_num();
		ASSERT(mp_lineGen != nullptr);
		mp_lineGen->at(currThread).setupGenerator(lor, n1, n2, *scanner);
	}

	for (int i_line = 0; i_line < numRaysToCast; i_line++)
	{
		unsigned int seed = 13;
		StraightLineParam randLine =
		    (i_line == 0) ? lor :
		                    mp_lineGen->at(currThread).getRandomLine(seed);
		randLine.point1 = randLine.point1 - offsetVec;
		randLine.point2 = randLine.point2 - offsetVec;

		double currentProjValue = 0.0;
		if (tofHelper != nullptr)
		{
			project_helper<true, true, true>(const_cast<Image*>(img),
			                                 randLine, currentProjValue,
			                                 tofHelper, tofValue);
		}
		else
		{
			project_helper<true, true, false>(const_cast<Image*>(img),
			                                  randLine, currentProjValue,
			                                  nullptr, 0);
		}
		imProj += currentProjValue;
	}

	if (numRaysToCast > 1)
	{
		imProj = imProj / static_cast<double>(numRaysToCast);
	}

	return imProj;
}

void GCOperatorProjectorSiddon::backProjection(
    Image* img, const StraightLineParam& lor, const Vector3D& n1,
    const Vector3D& n2, double projValue, const GCTimeOfFlightHelper* tofHelper,
    float tofValue) const
{
	const ImageParams& params = img->getParams();
	const Vector3D offsetVec = {params.off_x, params.off_y, params.off_z};


	int currThread = 0;
	double projValuePerLor = projValue;
	if (m_numRays > 1)
	{
		ASSERT(mp_lineGen != nullptr);
		currThread = omp_get_thread_num();
		mp_lineGen->at(currThread).setupGenerator(lor, n1, n2, *scanner);
		projValuePerLor = projValue / static_cast<double>(m_numRays);
	}

	for (int i_line = 0; i_line < m_numRays; i_line++)
	{
		unsigned int seed = 13;
		StraightLineParam randLine =
		    (i_line == 0) ? lor :
		                    mp_lineGen->at(currThread).getRandomLine(seed);
		randLine.point1 = randLine.point1 - offsetVec;
		randLine.point2 = randLine.point2 - offsetVec;
		if (tofHelper != nullptr)
		{
			project_helper<false, true, true>(img, randLine, projValuePerLor,
			                                  tofHelper, tofValue);
		}
		else
		{
			project_helper<false, true, false>(img, randLine, projValuePerLor,
			                                   nullptr, 0);
		}
	}
}

double GCOperatorProjectorSiddon::singleForwardProjection(
    const Image* img, const StraightLineParam& lor,
    const GCTimeOfFlightHelper* tofHelper, float tofValue)
{
	double v;
	project_helper<true, true, false>(const_cast<Image*>(img), lor, v,
	                                  tofHelper, tofValue);
	return v;
}

void GCOperatorProjectorSiddon::singleBackProjection(
    Image* img, const StraightLineParam& lor, double projValue,
    const GCTimeOfFlightHelper* tofHelper, float tofValue)
{
	project_helper<false, true, false>(img, lor, projValue, tofHelper,
	                                   tofValue);
}


enum SIDDON_DIR
{
	DIR_X = 0b001,
	DIR_Y = 0b010,
	DIR_Z = 0b100
};

// Note: FLAG_INCR skips the conversion from physical to logical coordinates by
// moving from pixel to pixel as the ray parameter is updated.  This may cause
// issues near the last intersection, which must therefore be handled with extra
// care.  Speedups around 20% were measured with FLAG_INCR=true.  Both versions
// are compared in tests, the "faster" version (FLAG_INCR=true) is used by
// default.
template <bool IS_FWD, bool FLAG_INCR, bool FLAG_TOF>
void GCOperatorProjectorSiddon::project_helper(
    Image* img, const StraightLineParam& lor, double& value,
    const GCTimeOfFlightHelper* tofHelper, float tofValue)
{
	if (IS_FWD)
	{
		value = 0.0;
	}

	ImageParams params = img->getParams();

	const Vector3D& p1 = lor.point1;
	const Vector3D& p2 = lor.point2;
	// 1. Intersection with FOV
	double t0;
	double t1;
	// Intersection with (centered) FOV cylinder
	double A = (p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y);
	double B = 2.0 * ((p2.x - p1.x) * p1.x + (p2.y - p1.y) * p1.y);
	double C =
	    p1.x * p1.x + p1.y * p1.y - params.fov_radius * params.fov_radius;
	double Delta = B * B - 4 * A * C;
	if (A != 0.0)
	{
		if (Delta <= 0.0)
		{
			t0 = 1.0;
			t1 = 0.0;
			return;
		}
		t0 = (-B - sqrt(Delta)) / (2 * A);
		t1 = (-B + sqrt(Delta)) / (2 * A);
	}
	else
	{
		t0 = 0.0;
		t1 = 1.0;
	}

	double d_norm = (p1 - p2).getNorm();
	bool flat_x = (p1.x == p2.x);
	bool flat_y = (p1.y == p2.y);
	bool flat_z = (p1.z == p2.z);
	double inv_p12_x = flat_x ? 0.0 : 1 / (p2.x - p1.x);
	double inv_p12_y = flat_y ? 0.0 : 1 / (p2.y - p1.y);
	double inv_p12_z = flat_z ? 0.0 : 1 / (p2.z - p1.z);
	int dir_x = (inv_p12_x >= 0.0) ? 1 : -1;
	int dir_y = (inv_p12_y >= 0.0) ? 1 : -1;
	int dir_z = (inv_p12_z >= 0.0) ? 1 : -1;

	// 2. Intersection with volume
	double dx = params.vx;
	double dy = params.vy;
	double dz = params.vz;
	double inv_dx = 1.0 / dx;
	double inv_dy = 1.0 / dy;
	double inv_dz = 1.0 / dz;

	double x0 = -params.length_x / 2;
	double x1 = params.length_x / 2;
	double y0 = -params.length_y / 2;
	double y1 = params.length_y / 2;
	double z0 = -params.length_z / 2;
	double z1 = params.length_z / 2;
	double ax_min, ax_max, ay_min, ay_max, az_min, az_max;
	get_alpha(-0.5 * params.length_x, 0.5 * params.length_x, p1.x, p2.x,
	          inv_p12_x, ax_min, ax_max);
	get_alpha(-0.5 * params.length_y, 0.5 * params.length_y, p1.y, p2.y,
	          inv_p12_y, ay_min, ay_max);
	get_alpha(-0.5 * params.length_z, 0.5 * params.length_z, p1.z, p2.z,
	          inv_p12_z, az_min, az_max);
	double amin = std::max({0.0, t0, ax_min, ay_min, az_min});
	double amax = std::min({1.0, t1, ax_max, ay_max, az_max});
	if (FLAG_TOF)
	{
		double amin_tof, amax_tof;
		tofHelper->getAlphaRange(amin_tof, amax_tof, d_norm, tofValue);
		amin = std::max(amin, amin_tof);
		amax = std::min(amax, amax_tof);
	}

	double a_cur = amin;
	double a_next = -1.0;
	double x_cur = (inv_p12_x > 0.0) ? x0 : x1;
	double y_cur = (inv_p12_y > 0.0) ? y0 : y1;
	double z_cur = (inv_p12_z > 0.0) ? z0 : z1;
	if ((inv_p12_x >= 0.0 && p1.x > x1) || (inv_p12_x < 0.0 && p1.x < x0) ||
	    (inv_p12_y >= 0.0 && p1.y > y1) || (inv_p12_y < 0.0 && p1.y < y0) ||
	    (inv_p12_z >= 0.0 && p1.z > z1) || (inv_p12_z < 0.0 && p1.z < z0))
	{
		return;
	}
	// Move starting point inside FOV
	double ax_next = flat_x ? std::numeric_limits<double>::max() : ax_min;
	if (!flat_x)
	{
		int kx = (int)ceil(dir_x * (a_cur * (p2.x - p1.x) - x_cur + p1.x) / dx);
		x_cur += kx * dir_x * dx;
		ax_next = (x_cur - p1.x) * inv_p12_x;
	}
	double ay_next = flat_y ? std::numeric_limits<double>::max() : ay_min;
	if (!flat_y)
	{
		int ky = (int)ceil(dir_y * (a_cur * (p2.y - p1.y) - y_cur + p1.y) / dy);
		y_cur += ky * dir_y * dy;
		ay_next = (y_cur - p1.y) * inv_p12_y;
	}
	double az_next = flat_z ? std::numeric_limits<double>::max() : az_min;
	if (!flat_z)
	{
		int kz = (int)ceil(dir_z * (a_cur * (p2.z - p1.z) - z_cur + p1.z) / dz);
		z_cur += kz * dir_z * dz;
		az_next = (z_cur - p1.z) * inv_p12_z;
	}
	// Pixel location (move pixel to pixel instead of calculating position for
	// each intersection)
	bool flag_first = true;
	int vx = -1;
	int vy = -1;
	int vz = -1;
	// The dir variables operate as binary bit-flags to determine in which
	// direction the current pixel should move: format 0bzyx (where z, y and x
	// are bits set to 1 when the pixel should move in the corresponding
	// direction, e.g. 0b101 moves in the z and x directions)
	short dir_prev = -1;
	short dir_next = -1;

	// Prepare data pointer (this assumes that the data is stored as a
	// contiguous array)
	double* raw_img_ptr = img->getData().getRawPointer();
	double* cur_img_ptr = nullptr;
	int num_x = params.nx;
	int num_xy = params.nx * params.ny;

	float ax_next_prev = ax_next;
	float ay_next_prev = ay_next;
	float az_next_prev = az_next;

	// 3. Integrate along ray
	bool flag_done = false;
	while (a_cur < amax && !flag_done)
	{
		// Find next intersection (along x, y or z)
		dir_next = 0b000;
		if (ax_next_prev <= ay_next_prev && ax_next_prev <= az_next_prev)
		{
			a_next = ax_next;
			x_cur += dir_x * dx;
			ax_next = (x_cur - p1.x) * inv_p12_x;
			dir_next |= SIDDON_DIR::DIR_X;
		}
		if (ay_next_prev <= ax_next_prev && ay_next_prev <= az_next_prev)
		{
			a_next = ay_next;
			y_cur += dir_y * dy;
			ay_next = (y_cur - p1.y) * inv_p12_y;
			dir_next |= SIDDON_DIR::DIR_Y;
		}
		if (az_next_prev <= ax_next_prev && az_next_prev <= ay_next_prev)
		{
			a_next = az_next;
			z_cur += dir_z * dz;
			az_next = (z_cur - p1.z) * inv_p12_z;
			dir_next |= SIDDON_DIR::DIR_Z;
		}
		// Clip to FOV range
		if (a_next > amax)
		{
			a_next = amax;
		}
		if (a_cur >= a_next)
		{
			ax_next_prev = ax_next;
			ay_next_prev = ay_next;
			az_next_prev = az_next;
			continue;
		}
		// Determine pixel location
		float tof_weight = 1.f;
		double a_mid = 0.5 * (a_cur + a_next);
		if (FLAG_TOF)
		{
			tof_weight = tofHelper->getWeight(d_norm, tofValue, a_cur * d_norm,
			                                  a_next * d_norm);
		}
		if (!FLAG_INCR || flag_first)
		{
			vx = (int)((p1.x + a_mid * (p2.x - p1.x) + params.length_x / 2) *
			           inv_dx);
			vy = (int)((p1.y + a_mid * (p2.y - p1.y) + params.length_y / 2) *
			           inv_dy);
			vz = (int)((p1.z + a_mid * (p2.z - p1.z) + params.length_z / 2) *
			           inv_dz);
			cur_img_ptr = raw_img_ptr + vz * num_xy + vy * num_x;
			flag_first = false;
			if (vx < 0 || vx >= params.nx || vy < 0 || vy >= params.ny ||
			    vz < 0 || vz >= params.nz)
			{
				flag_done = true;
			}
		}
		else
		{
			if (dir_prev & SIDDON_DIR::DIR_X)
			{
				vx += dir_x;
				if (vx < 0 || vx >= params.nx)
				{
					flag_done = true;
				}
			}
			if (dir_prev & SIDDON_DIR::DIR_Y)
			{
				vy += dir_y;
				if (vy < 0 || vy >= params.ny)
				{
					flag_done = true;
				}
				else
				{
					cur_img_ptr += dir_y * num_x;
				}
			}
			if (dir_prev & SIDDON_DIR::DIR_Z)
			{
				vz += dir_z;
				if (vz < 0 || vz >= params.nz)
				{
					flag_done = true;
				}
				else
				{
					cur_img_ptr += dir_z * num_xy;
				}
			}
		}
		if (flag_done)
		{
			continue;
		}
		dir_prev = dir_next;
		double weight = (a_next - a_cur) * d_norm;
		if (FLAG_TOF)
		{
			weight *= tof_weight;
		}
		if (IS_FWD)
		{
			value += weight * cur_img_ptr[vx];
		}
		else
		{
			double output = value * weight;
			double* ptr = &cur_img_ptr[vx];
#pragma omp atomic
			*ptr += output;
		}
		a_cur = a_next;
		ax_next_prev = ax_next;
		ay_next_prev = ay_next;
		az_next_prev = az_next;
	}
}


// Explicit instantiation of slow version used in tests
template void GCOperatorProjectorSiddon::project_helper<true, false, true>(
    Image* img, const StraightLineParam&, double&,
    const GCTimeOfFlightHelper*, float);
template void GCOperatorProjectorSiddon::project_helper<false, false, true>(
    Image* img, const StraightLineParam&, double&,
    const GCTimeOfFlightHelper*, float);
template void GCOperatorProjectorSiddon::project_helper<true, false, false>(
    Image* img, const StraightLineParam&, double&,
    const GCTimeOfFlightHelper*, float);
template void GCOperatorProjectorSiddon::project_helper<false, false, false>(
    Image* img, const StraightLineParam&, double&,
    const GCTimeOfFlightHelper*, float);
