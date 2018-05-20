#ifndef KDTREE_H
#define KDTREE_H

#include "Ray.h"
#include "Triangle.h"
#include <LinearMath\btVector3.h>
#include <vector>
#include <assert.h>

class KDNode
{
public:
	KDNode() {};
	KDNode* Build(std::vector<Triangle*> &triangles_, int depth);
	bool Hit(KDNode* node, const Ray &ray, double &t, double &tmin, btVector3 &normal, Material &m, btTransform transform);

	AABBox box;
	KDNode* left;
	KDNode* right;
	std::vector<Triangle*> triangles;
	bool leaf;

};

#endif