/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/StraightLineParam.hpp"
#include "geometry/Vector3D.hpp"

class Plane
{

public:
	Vector3D point1;
	Vector3D point2;
	Vector3D point3;
	Vector3D dir;
	double a;
	double b;
	double c;
	double d;
	// equation of the plane: a.x + b.y + c.z + d = 0
public:
	Plane();
	Plane(const Vector3D& pt1, const Vector3D& pt2, const Vector3D& pt3);
	void update(const Vector3D& pt1, const Vector3D& pt2, const Vector3D& pt3);
	void update_eq(const Vector3D& pt1, const Vector3D& pt2, const Vector3D& pt3);
	bool isCoplanar(const Vector3D& pt) const;
	bool isParrallel(const StraightLineParam& line) const;
	Vector3D findInterLine(const StraightLineParam& line) const;
};
