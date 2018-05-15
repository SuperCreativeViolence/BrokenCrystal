#include "Material.h"

Material::Material(MaterialType t, const btVector3 & c, const btVector3 & e, Texture tex)
{
	m_type = t, m_colour = c, m_emission = e;
	m_texture = tex;
}

MaterialType Material::get_type() const
{
	return m_type;
}
btVector3 Material::get_colour() const
{
	return m_colour;
}

btVector3 Material::get_colour_at(double u, double v) const
{
	if (m_texture.is_loaded())
		return m_texture.get_pixel(u, v);

	return m_colour;
}

btVector3 Material::get_emission() const
{
	return m_emission;
}

Ray Material::get_reflected_ray(const Ray & r, btVector3 & p, const btVector3 & n, unsigned short * Xi) const
{
	double roughness = 0.8;
	// Ideal specular reflection
	if (m_type == SPEC)
	{
		double roughness = 0.8;
		btVector3 reflected = r.direction - n * 2 * n.dot(r.direction);
		reflected = btVector3(
			reflected.x() + (erand48(Xi) - 0.5)*roughness,
			reflected.y() + (erand48(Xi) - 0.5)*roughness,
			reflected.z() + (erand48(Xi) - 0.5)*roughness
		).normalize();

		return Ray(p, reflected);
	}
	// Ideal diffuse reflection
	if (m_type == DIFF)
	{
		btVector3 nl = n.dot(r.direction) < 0 ? n : n * -1;
		double r1 = 2 * M_PI*erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
		btVector3 w = nl;
		btVector3 u = ((fabs(w.x()) > .1 ? btVector3(0, 1, 0) : btVector3(1, 0, 0)).cross(w)).normalize();
		btVector3 v = w.cross(u);
		btVector3 d = (u*cos(r1)*r2s + v * sin(r1)*r2s + w * sqrt(1 - r2)).normalize();

		return Ray(p, d);
	}
}
