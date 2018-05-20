#include "Object.h"

ObjectIntersection::ObjectIntersection(bool hit_, double u_, const btVector3& normal_, Material material_)
{
	hit = hit_;
	u = u_;
	normal = normal_;
	material = material_;
}

Object::Object(btCollisionShape* pShape, const btVector3 &position, const btQuaternion &rotation, Material material_, float mass)
{
	shape = pShape;

	btTransform transform = btTransform::getIdentity();
	transform.setOrigin(position);
	transform.setRotation(rotation);
	motionState = new OpenglMotionState(transform);

	btVector3 localInteria(0, 0, 0);
	if (mass != 0.0f)
		pShape->calculateLocalInertia(mass, localInteria);

	btRigidBody::btRigidBodyConstructionInfo cInfo(mass, motionState, pShape, localInteria);
	body = new btRigidBody(cInfo);
	material = material_;
}

Object::~Object()
{
	delete body;
	delete motionState;
	delete shape;
}

Sphere::Sphere(const btVector3 &position_, double radius_, float mass_, Material material_) : Object(new btSphereShape(radius_), position_, btQuaternion(1, 0, 0, 0), material_, mass_)
{
	radius = radius_;
}

ObjectIntersection Sphere::GetIntersection(const Ray& ray)
{
	bool hit = false;
	double distance = 0;
	btVector3 normal = btVector3(0, 0, 0);
	btVector3 position = GetPosition();

	btVector3 op = position - ray.origin;
	double t;
	double b = op.dot(ray.direction);
	double det = b * b - op.dot(op) + radius * radius;

	if (det < 0) return ObjectIntersection(hit, distance, normal, material);
	else det = sqrt(det);

	distance = (t = b - det) > EPSILON ? t : ((t = b + det) > EPSILON ? t : 0);
	if (distance != 0)
	{
		hit = true;
		normal = ((ray.origin + ray.direction * distance) - position).normalize();
	}
	return ObjectIntersection(hit, distance, normal, material);
}

Mesh::Mesh(const btVector3 & position_, std::vector<Triangle*> triangles_, float mass, Material material_) : Object(new btEmptyShape(), position_, btQuaternion(1, 0, 0, 0), material_, mass)
{
	triangles = triangles_;
	btTriangleMesh* triangleMesh = new btTriangleMesh();
	for (auto & triangle : triangles)
	{
		triangleMesh->addTriangle(triangle->pos[0], triangle->pos[1], triangle->pos[2]);
	}

	// KDTree
	node = KDNode().Build(triangles, 0);

	btConvexShape* tempShape = new btConvexTriangleMeshShape(triangleMesh);
	btShapeHull* hull = new btShapeHull(tempShape);
	btScalar margin = tempShape->getMargin();
	hull->buildHull(margin);
	tempShape->setUserPointer(hull);
	shape = tempShape;
	btVector3 localInteria(0, 0, 0);
	if (mass != 0.0f)
		shape->calculateLocalInertia(mass, localInteria);

	btRigidBody::btRigidBodyConstructionInfo cInfo(mass, motionState, shape, localInteria);
	body = new btRigidBody(cInfo);
}

Mesh::Mesh(const btVector3& position_, const char* filePath, float mass, Material material_) : Object(new btEmptyShape(), position_, btQuaternion(1, 0, 0, 0))
{
	std::string mtlBasePath;
	std::string inputFile = filePath;
	unsigned long pos = inputFile.find_last_of("/");
	mtlBasePath = inputFile.substr(0, pos + 1);

	std::vector<tinyobj::shape_t> obj_shapes;
	std::vector<tinyobj::material_t> obj_materials;

	//std::vector<Material> materials;

	printf("Loading %s...\n", filePath);
	std::string err = tinyobj::LoadObj(obj_shapes, obj_materials, inputFile.c_str(), mtlBasePath.c_str());

	if (!err.empty())
	{
		std::cerr << err << std::endl;
	}

	//for (int i = 0; i < obj_materials.size(); i++)
	//{
	//	std::string texturePath = "";

	//	if (!obj_materials[i].diffuse_texname.empty())
	//	{
	//		if (obj_materials[i].diffuse_texname[0] == '/') texturePath= obj_materials[i].diffuse_texname;
	//		texturePath = mtlBasePath + obj_materials[i].diffuse_texname;
	//		materials.push_back(Material(DIFF, btVector3(1, 1, 1), btVector3(), texturePath.c_str()));
	//	}
	//	else
	//	{
	//		materials.push_back(Material(DIFF, btVector3(1, 1, 1), btVector3()));
	//	}
	//}

	long shapeSize, indicesSize;
	shapeSize = obj_shapes.size();

	for (int i = 0; i < shapeSize; i++)
	{
		indicesSize = obj_shapes[i].mesh.indices.size() / 3;
		for (size_t f = 0; f < indicesSize; f++)
		{

			// Triangle vertex coordinates
			btVector3 v0_ = btVector3(
				obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f] * 3],
				obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f] * 3 + 1],
				obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f] * 3 + 2]
			);

			btVector3 v1_ = btVector3(
				obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f + 1] * 3],
				obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f + 1] * 3 + 1],
				obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f + 1] * 3 + 2]
			);

			btVector3 v2_ = btVector3(
				obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f + 2] * 3],
				obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f + 2] * 3 + 1],
				obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f + 2] * 3 + 2]
			);

			btVector3 t0_, t1_, t2_;

			//Attempt to load triangle texture coordinates
			if (obj_shapes[i].mesh.indices[3 * f + 2] * 2 + 1 < obj_shapes[i].mesh.texcoords.size())
			{
				t0_ = btVector3(
					obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f] * 2],
					obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f] * 2 + 1],
					0
				);

				t1_ = btVector3(
					obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f + 1] * 2],
					obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f + 1] * 2 + 1],
					0
				);

				t2_ = btVector3(
					obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f + 2] * 2],
					obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f + 2] * 2 + 1],
					0
				);
			}
			else
			{
				t0_ = btVector3();
				t1_ = btVector3();
				t2_ = btVector3();
			}

			//if (obj_shapes[i].mesh.material_ids[f] < materials.size())
			//	triangles.push_back(new Triangle(v0_, v1_, v2_, t0_, t1_, t2_, materials[obj_shapes[i].mesh.material_ids[f]]));
			//else
			//	triangles.push_back(new Triangle(v0_, v1_, v2_, t0_, t1_, t2_, material));
			triangles.push_back(new Triangle(v0_, v1_, v2_, t0_, t1_, t2_, material_));
		}
	}

	btTriangleMesh* triangleMesh = new btTriangleMesh();
	for (auto & triangle : triangles)
	{
		triangleMesh->addTriangle(triangle->pos[0], triangle->pos[1], triangle->pos[2]);
	}

	// KDTree
	node = KDNode().Build(triangles, 0);

	btConvexShape* tempShape = new btConvexTriangleMeshShape(triangleMesh);
	btShapeHull* hull = new btShapeHull(tempShape);
	btScalar margin = tempShape->getMargin();
	hull->buildHull(margin);
	tempShape->setUserPointer(hull);
	shape = tempShape;
	btVector3 localInteria(0, 0, 0);
	if (mass != 0.0f)
		shape->calculateLocalInertia(mass, localInteria);

	btRigidBody::btRigidBodyConstructionInfo cInfo(mass, motionState, shape, localInteria);
	body = new btRigidBody(cInfo);

	obj_shapes.clear();
	obj_materials.clear();
}

ObjectIntersection Mesh::GetIntersection(const Ray& ray)
{
#ifdef USE_KDTREE
	double t = 0, tmin = INFINITY;
	btVector3 normal = btVector3(0,0,0);
	Material material = Material();
	bool hit = node->Hit(node, ray, t, tmin, normal, material, body->getWorldTransform());
	return ObjectIntersection(hit, tmin, normal, material);
#else
	float tNear = std::numeric_limits<float>::max();
	btTransform transform = body->getWorldTransform();
	ObjectIntersection intersection = ObjectIntersection();
	for (auto & triangle : triangles)
	{
		float u, v;
		ObjectIntersection temp = triangle->GetIntersection(ray, transform);
		if (temp.hit && temp.u < tNear)
		{
			tNear = temp.u;
			intersection = temp;
		}
	}
	return intersection;
#endif
}

std::vector<Triangle*> Mesh::GetTriangles() const
{
	return triangles;
}
