#ifndef VORONOI_H
#define VORONOI_H

#include <math.h>
#include <time.h>
#include <vector>
#include "Triangle.h"
#include "Object.h"

std::vector<Mesh*> break_into_pieces(Mesh* mesh, int pieces);
std::vector<std::vector<Triangle*>*>* voronoi_Fracture(std::vector<Triangle*> triangles);

#endif
