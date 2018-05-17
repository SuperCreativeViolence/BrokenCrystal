#include "erand48.h"

std::default_random_engine generator;
std::uniform_real_distribution<double> distr(0.0, 1.0);

void _dorand48(unsigned short xseed[3])
{
	unsigned long accu;
	unsigned short temp[2];

	accu = (unsigned long) RAND48_MULT_0 * (unsigned long) xseed[0] +
		(unsigned long) RAND48_ADD;
	temp[0] = (unsigned short) accu;        /* lower 16 bits */
	accu >>= sizeof(unsigned short) * 8;
	accu += (unsigned long) RAND48_MULT_0 * (unsigned long) xseed[1] +
		(unsigned long) RAND48_MULT_1 * (unsigned long) xseed[0];
	temp[1] = (unsigned short) accu;        /* middle 16 bits */
	accu >>= sizeof(unsigned short) * 8;
	accu += RAND48_MULT_0 * xseed[2] + RAND48_MULT_1 * xseed[1] + RAND48_MULT_2 * xseed[0];
	xseed[0] = temp[0];
	xseed[1] = temp[1];
	xseed[2] = (unsigned short) accu;
}

double erand48(unsigned short xseed[3])
{
	//_dorand48(xseed);
	//return ldexp((double) xseed[0], -48) +
	//	ldexp((double) xseed[1], -32) +
	//	ldexp((double) xseed[2], -16);
	return distr(generator);
}
