#ifndef IMAGEDENOISER_
#define IMAGEDENOISER_

# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <fstream>
# include <ctime>
# include <cstring>
#include <vector>

class ImageDenoiser
{
public:
	std::vector<unsigned char> gray_median_news(int m, int n, std::vector<unsigned char> gray);

private:
	int i4vec_frac(int n, int* a, int k);
	int i4vec_median(int n, int* a);
};

#endif