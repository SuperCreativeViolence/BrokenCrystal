#include "erand48.h"

std::default_random_engine generator;
std::uniform_real_distribution<double> distr(0.0, 1.0);

double erand48()
{
	return distr(generator);
}