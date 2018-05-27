#ifndef VORONOI_H
#define VORONOI_H

#include <math.h>
#include <time.h>
#include <vector>
#include "Triangle.h"
#include "Object.h"

std::vector<Mesh*> break_into_pieces(Mesh* mesh, int pieces);
std::vector<Mesh*> break_into_pieces2(Mesh* mesh, int pieces);
std::vector<std::vector<Triangle*>> voronoi_Fracture(std::vector<Triangle*> triangles);
float get_mesh_mass(std::vector<Triangle*> triangles, btVector3 position);

#endif
