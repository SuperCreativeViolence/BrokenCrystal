#include "KDTree.h"

KDNode* KDNode::Build(std::vector<Triangle*> &triangles_, int depth)
{
	KDNode* node = new KDNode();
	node->leaf = false;
	node->triangles = std::vector<Triangle*>();
	node->left = NULL;
	node->right = NULL;
	node->box = AABBox();

	if (triangles_.size() == 0) return node;

	if (depth > 25 || triangles_.size() <= 6)
	{
		node->triangles = triangles_;
		node->leaf = true;
		node->box = triangles_[0]->GetBoundingBox();

		for (long i = 1; i < triangles_.size(); i++)
		{
			node->box.Expand(triangles_[i]->GetBoundingBox());
		}

		node->left = new KDNode();
		node->right = new KDNode();
		node->left->triangles = std::vector<Triangle*>();
		node->right->triangles = std::vector<Triangle*>();

		return node;
	}

	node->box = triangles_[0]->GetBoundingBox();
	btVector3 midpt = btVector3(0, 0, 0);
	double trianglesP = 1.0 / triangles_.size();

	for (long i = 1; i < triangles_.size(); i++)
	{
		node->box.Expand(triangles_[i]->GetBoundingBox());
		midpt = midpt + (triangles_[i]->GetMidPoint() * trianglesP);
	}

	std::vector<Triangle*> leftTriangles;
	std::vector<Triangle*> rightTriangles;
	int axis = node->box.GetLongestAxis();

	for (long i = 0; i < triangles_.size(); i++)
	{
		switch (axis)
		{
			case 0:
				midpt[0] >= triangles_[i]->GetMidPoint()[0] ? rightTriangles.push_back(triangles_[i]) : leftTriangles.push_back(triangles_[i]);
				break;
			case 1:
				midpt[1] >= triangles_[i]->GetMidPoint()[1] ? rightTriangles.push_back(triangles_[i]) : leftTriangles.push_back(triangles_[i]);
				break;
			case 2:
				midpt[2] >= triangles_[i]->GetMidPoint()[2] ? rightTriangles.push_back(triangles_[i]) : leftTriangles.push_back(triangles_[i]);
				break;
		}
	}

	if (triangles_.size() == leftTriangles.size() || triangles_.size() == rightTriangles.size())
	{
		node->triangles = triangles_;
		node->leaf = true;
		node->box = triangles_[0]->GetBoundingBox();

		for (long i = 1; i < triangles_.size(); i++)
		{
			node->box.Expand(triangles_[i]->GetBoundingBox());
		}

		node->left = new KDNode();
		node->right = new KDNode();
		node->left->triangles = std::vector<Triangle*>();
		node->right->triangles = std::vector<Triangle*>();

		return node;
	}

	node->left = Build(leftTriangles, depth + 1);
	node->right = Build(rightTriangles, depth + 1);

	return node;
}

bool KDNode::Hit(KDNode* node, const Ray &ray, double &t, double &tmin, btVector3 &normal, Material& material, btTransform transform)
{
	double dist;
	assert(node->box);
	if (node->box.Intersection(ray, dist, transform))
	{
		if (dist > tmin) return false;

		bool hitTriangle = false;
		bool hitLeft = false;
		bool hitRight = false;
		long triangleIndex;

		if (!node->leaf)
		{
			hitLeft = Hit(node->left, ray, t, tmin, normal, material, transform);
			hitRight = Hit(node->right, ray, t, tmin, normal, material, transform);
			return hitLeft || hitRight;
		}
		else
		{
			long triangles_size = node->triangles.size();
			for (long i = 0; i < triangles_size; i++)
			{
				if (node->triangles[i]->Intersect(ray, t, tmin, normal, transform))
				{
					hitTriangle = true;
					tmin = t;
					triangleIndex = i;
				}
			}
			if (hitTriangle)
			{
				btVector3 p = ray.origin + ray.direction * tmin;
				btVector3 color = node->triangles[triangleIndex]->GetColorAt(p);
				material = node->triangles[triangleIndex]->GetMaterial();
				material.SetColor(color);
				return true;
			}
		}
	}
	return false;
}
