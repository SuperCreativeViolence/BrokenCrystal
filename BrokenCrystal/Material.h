#ifndef MATERIAL_H
#define MATERIAL_H

#include "erand48.h"
#include "Ray.h"
#include "Texture.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum MaterialType
{
	DIFF, SPEC, EMIT
};


class Material
{

private:
	MaterialType m_type;
	btVector3 m_colour;
	btVector3 m_emission;
	Texture m_texture;


public:
	Material(MaterialType t = DIFF, const btVector3 & c = btVector3(1, 1, 1), const btVector3 & e = btVector3(0, 0, 0), Texture tex = Texture());
	MaterialType get_type() const;
	btVector3 get_colour() const;
	btVector3 get_colour_at(double u, double v) const;
	btVector3 get_emission() const;
	Ray get_reflected_ray(const Ray &r, btVector3 &p, const btVector3 &n, unsigned short *Xi) const;

};


#endif 