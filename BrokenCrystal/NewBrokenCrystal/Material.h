#ifndef MATERIAL_H
#define MATERIAL_H

#include "erand48.h"
#include "Material.h"
#include "Ray.h"
#include <LinearMath\btVector3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum MaterialType { DIFF, SPEC, EMIT };

class Material
{
public:
	Material(MaterialType type_ = DIFF, btVector3 color_ = btVector3(1, 1, 1), btVector3 emission_ = btVector3(0, 0, 0));
	MaterialType GetType() const;
	btVector3 GetColor() const;
	btVector3 GetEmission() const;
	Ray GetReflectedRay(const Ray &ray, const btVector3 &position, const btVector3 &normal, unsigned short *Xi) const;

private:
	MaterialType type;
	btVector3 color;
	btVector3 emission;
};

#endif