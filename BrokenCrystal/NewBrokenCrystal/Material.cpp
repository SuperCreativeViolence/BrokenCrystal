#include "Material.h"

Material::Material(MaterialType type_, btVector3 color_, btVector3 emission_, Texture* texture_)
{
	type = type_;
	color = color_;
	emission = emission_;
	texture = texture_;
}

MaterialType Material::GetType() const
{
	return type;
}

btVector3 Material::GetColor() const
{
	return color;
}

btVector3 Material::GetColorAt(double u, double v) const
{
	if (texture != nullptr && texture->IsLoaded())
		return texture->GetPixel(u, v);

	return color;
}

btVector3 Material::GetEmission() const
{
	return emission;
}

Ray Material::GetReflectedRay(const Ray & ray, const btVector3 & position, const btVector3 & normal, btVector3 &color) const
{
	
	if (type == SPEC)
	{
		btVector3 reflected = ray.direction - normal * 2 * normal.dot(ray.direction);
		return Ray(position, reflected);
	}
	else if (type == GLOSS)
	{
		double roughness = 0.7;
		btVector3 reflected = ray.direction - normal * 2 * normal.dot(ray.direction);
		reflected = btVector3(
			reflected[0] + (erand48() - 0.5) * roughness,
			reflected[1] + (erand48() - 0.5) * roughness,
			reflected[2] + (erand48() - 0.5) * roughness
		).normalize();

		return Ray(position, reflected);
	}
	else if (type == DIFF)
	{
		btVector3 nl = normal.dot(ray.direction) < 0 ? normal : normal * -1;
		double r1 = 2 * M_PI * erand48();
		double r2 = erand48();
		double r2s = sqrt(r2);

		btVector3 w = nl;
		btVector3 u;
		if (fabs(w[0]) > 0.1)
			u = (btVector3(0.0, 1.0, 0.0).cross(w)).normalize();
		else
			u = (btVector3(1.0, 0.0, 0.0).cross(w)).normalize();
		btVector3 v = w.cross(u);
		btVector3 d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).normalize();
		return Ray(position, d);
	}
	else if (type == TRANS)
	{
		btVector3 nl = normal.dot(ray.direction) < 0 ? normal : normal * -1;
		btVector3 reflection = ray.direction - normal * 2 * normal.dot(ray.direction);
		bool into = normal.dot(nl) > 0;		  
		double nc = 1;						  
		double nt = 1.5;					  
		double nnt;

		double Re, RP, TP, Tr;
		btVector3 tdir = btVector3(0,0,0);

		if (into)	  
			nnt = nc / nt;
		else
			nnt = nt / nc;

		double ddn = ray.direction.dot(nl);
		double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);

		if (cos2t < 0) return Ray(position, reflection);

		if (into)
			tdir = (ray.direction * nnt - normal * (ddn * nnt + sqrt(cos2t))).normalize();
		else
			tdir = (ray.direction * nnt + normal * (ddn * nnt + sqrt(cos2t))).normalize();

		double a = nt - nc;
		double b = nt + nc;
		double R0 = a * a / (b * b);

		double c;
		if (into)
			c = 1 + ddn;
		else
			c = 1 - tdir.dot(normal);

		Re = R0 + (1 - R0) * c * c * c * c * c;	 
		Tr = 1 - Re;						

		double P = .25 + .5 * Re;
		RP = Re / P;			
		TP = Tr / (1 - P);

		if (erand48() < P)
		{
			color = color * (RP);
			return Ray(position, reflection);
		}

		color = color * (TP);
		return Ray(position, tdir);
	}
	else
	{
		printf("Material Type Error\n");
		exit(0);
	}
}

void Material::SetColor(const btVector3& color)
{
	this->color = color;
}


float3 Material::GetColorF()
{
	return make_float3(color.getX(), color.getY(), color.getZ());
}
float3 Material::GetEmissionF()
{
	return make_float3(emission.getX(), emission.getY(), emission.getZ());
}
