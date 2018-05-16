#ifndef ERAND48_H
#define ERAND48_H
#include <math.h>
#include <stdlib.h>

#define RAND48_SEED_0   (0x330e)
#define RAND48_SEED_1   (0xabcd)
#define RAND48_SEED_2   (0x1234)
#define RAND48_MULT_0   (0xe66d)
#define RAND48_MULT_1   (0xdeec)
#define RAND48_MULT_2   (0x0005)
#define RAND48_ADD      (0x000b)

void _dorand48(unsigned short xseed[3]);
double erand48(unsigned short xseed[3]);
#endif