#ifndef AABBOX_H
#define AABBOX_H

#define EPSILON 1e-10


struct AABBox
{
	btVector3 min, max;

	AABBox(btVector3 min_ = btVector3(0, 0, 0), btVector3 max_ = btVector3(0, 0, 0))
	{
		min = min_;
		max = max_;
	}

	void Expand(const AABBox &box)
	{
		if (box.min.x() < min.x()) min[0] = box.min.x();
		if (box.min.y() < min.y()) min[1] = box.min.y();
		if (box.min.z() < min.z()) min[2] = box.min.z();

		if (box.max.x() > max.x()) max[0] = box.max.x();
		if (box.max.y() > max.y()) max[1] = box.max.y();
		if (box.max.z() > max.z()) max[2] = box.max.z();
	}

	// Expand to fit point
	void Expand(const btVector3 &vec)
	{
		if (vec.x() < min.x()) min[0] = vec.x();
		if (vec.y() < min.y()) min[1] = vec.y();
		if (vec.z() < min.z()) min[2] = vec.z();
	}

	// Returns longest axis: 0, 1, 2 for x, y, z respectively
	int GetLongestAxis()
	{
		btVector3 diff = max - min;
		if (diff.x() > diff.y() && diff.x() > diff.z()) return 0;
		if (diff.y() > diff.x() && diff.y() > diff.z()) return 1;
		return 2;
	}

	// Check if ray intersects with box. Returns true/false and stores distance in t
	bool Intersection(const Ray &r, double &t, btTransform transform)
	{
		btVector3 transMin = transform * min;
		btVector3 transMax = transform * max;

		double tx1 = (transMin.x() - r.origin.x())*r.direction_inv.x();
		double tx2 = (transMax.x() - r.origin.x())*r.direction_inv.x();

		double tmin = std::min(tx1, tx2);
		double tmax = std::max(tx1, tx2);

		double ty1 = (transMin.y() - r.origin.y())*r.direction_inv.y();
		double ty2 = (transMax.y() - r.origin.y())*r.direction_inv.y();

		tmin = std::max(tmin, std::min(ty1, ty2));
		tmax = std::min(tmax, std::max(ty1, ty2));

		double tz1 = (transMin.z() - r.origin.z())*r.direction_inv.z();
		double tz2 = (transMax.z() - r.origin.z())*r.direction_inv.z();

		tmin = std::max(tmin, std::min(tz1, tz2));
		tmax = std::min(tmax, std::max(tz1, tz2));
		t = tmin;

		return tmax >= tmin;
	}
};


#endif
