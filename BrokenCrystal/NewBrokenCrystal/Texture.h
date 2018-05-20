#ifndef TEXTURE_H
#define TEXTURE_H

#include <vector>
#include <LinearMath\btVector3.h>

class Texture
{
public:
	Texture(const char* fileName);
	Texture() {};
	btVector3 GetPixel(unsigned x, unsigned y) const;
	btVector3 GetPixel(double u, double v) const;
	bool IsLoaded() const;

private:
	unsigned width, height;
	bool loaded = false;
	std::vector<unsigned char> image;
	
};

#endif