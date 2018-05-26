#ifndef MATERIAL_H
#define MATERIAL_H

#include "Texture.h"
#include "erand48.h"
#include "tiny_obj_loader.h"
#include "Ray.h"
#include <LinearMath\btVector3.h>
#include "cutil_math.h"
#include "MaterialType.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


class Material
{
public:
	Material(MaterialType type_ = DIFF, btVector3 color_ = btVector3(1, 1, 1), btVector3 emission_ = btVector3(0, 0, 0), Texture* texture_ = nullptr);
	MaterialType GetType() const;
	btVector3 GetColor() const;
	btVector3 GetColorAt(double u, double v) const;
	btVector3 GetEmission() const;
	Ray GetReflectedRay(const Ray &ray, const btVector3 &position, const btVector3 &normal, btVector3 &color) const;
	void SetColor(const btVector3& color);
	float3 GetColorF();
	float3 GetEmissionF();

private:
	MaterialType type;
	btVector3 color;
	btVector3 emission;
	Texture* texture;
};

#endif