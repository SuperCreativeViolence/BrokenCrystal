#include "Material.h"

Material::Material(MaterialType type_, btVector3 color_, btVector3 emission_)
{
	type = type_;
	color = color_;
	emission = emission_;
}

MaterialType Material::GetType() const
{
	return type;
}

btVector3 Material::GetColor() const
{
	return color;
}

btVector3 Material::GetEmission() const
{
	return emission;
}

Ray Material::GetReflectedRay(const Ray & ray, const btVector3 & position, const btVector3 & normal, unsigned short * Xi = nullptr) const
{
	if (type == SPEC)
	{
		double roughness = 0.8;
		btVector3 reflected = ray.direction - normal * 2 * normal.dot(ray.direction);
		reflected = btVector3(
			reflected[0] + (erand48() - 0.5) * roughness,
			reflected[1] + (erand48() - 0.5) * roughness,
			reflected[2] + (erand48() - 0.5) * roughness
		).normalize();

		return Ray(position, reflected);
	}
	else
	{
		btVector3 nl = normal.dot(ray.direction) < 0 ? normal : normal * -1;
		double r1 = 2 * M_PI * erand48();
		double r2 = erand48();
		double r2s = sqrt(r2);

		btVector3 w = nl;
		btVector3 u = (fabs(w[0]) > 0.1 ? btVector3(0, 1, 0) : btVector3(1, 0, 0)).cross(w).normalize();
		btVector3 v = w.cross(u);
		btVector3 d = (u*cos(r1)*r2s + v * sin(r1)*r2s + w * sqrt(1 - r2)).normalize();

		return Ray(position, d);
	}
}
