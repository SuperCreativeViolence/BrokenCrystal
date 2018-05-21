#include "ImageDenoiser.h"

std::vector<unsigned char> ImageDenoiser::gray_median_news(int m, int n, std::vector<unsigned char> gray)
{
	std::vector<unsigned char> gray2 = std::vector<unsigned char>(m*n*4);
	int i;
	int j;
	int p[5];

	for (i = 1; i < m - 1; i++)
	{
		for (j = 1; j < n - 1; j++)
		{
			p[0] = gray[i - 1 + j * m];
			p[1] = gray[i + 1 + j * m];
			p[2] = gray[i + (j + 1)*m];
			p[3] = gray[i + (j - 1)*m];
			p[4] = gray[i + j * m];

			gray2[i + j * m] = i4vec_median(5, p);
		}
	}

	for (i = 1; i < m - 1; i++)
	{
		j = 0;
		p[0] = gray[i - 1 + j * m];
		p[1] = gray[i + 1 + j * m];
		p[2] = gray[i + j * m];
		p[3] = gray[i + (j + 1)*m];
		p[4] = gray[i + (j + 2)*m];
		gray2[i + j * m] = i4vec_median(5, p);

		j = n - 1;
		p[0] = gray[i - 1 + j * m];
		p[1] = gray[i + 1 + j * m];
		p[2] = gray[i + (j - 2)*m];
		p[3] = gray[i + (j - 1)*m];
		p[4] = gray[i + j * m];
		gray2[i + j * m] = i4vec_median(5, p);
	}

	for (j = 1; j < n - 1; j++)
	{
		i = 0;
		p[0] = gray[i + j * m];
		p[1] = gray[i + 1 + j * m];
		p[2] = gray[i + 2 + j * m];
		p[3] = gray[i + (j - 1)*m];
		p[4] = gray[i + (j + 1)*m];
		gray2[i + j * m] = i4vec_median(5, p);

		i = m - 1;
		p[0] = gray[i - 2 + j * m];
		p[1] = gray[i - 1 + j * m];
		p[2] = gray[i + j * m];
		p[3] = gray[i + (j - 1)*m];
		p[4] = gray[i + (j + 1)*m];
		gray2[i + j * m] = i4vec_median(5, p);
	}

	i = 0;
	j = 0;
	p[0] = gray[i + 1 + j * m];
	p[1] = gray[i + j * m];
	p[2] = gray[i + (j + 1)*m];
	gray2[i + j * m] = i4vec_median(3, p);

	i = 0;
	j = n - 1;
	p[0] = gray[i + 1 + j * m];
	p[1] = gray[i + j * m];
	p[2] = gray[i + (j - 1)*m];
	gray2[i + j * m] = i4vec_median(3, p);

	i = m - 1;
	j = 0;
	p[0] = gray[i - 1 + j * m];
	p[1] = gray[i + j * m];
	p[2] = gray[i + (j + 1)*m];
	gray2[i + j * m] = i4vec_median(3, p);

	i = m - 1;
	j = n - 1;
	p[0] = gray[i - 1 + j * m];
	p[1] = gray[i + j * m];
	p[2] = gray[i + (j - 1)*m];
	gray2[i + j * m] = i4vec_median(3, p);

	return gray2;
}

int ImageDenoiser::i4vec_frac(int n, int* a, int k)
{
	int frac;
	int i;
	int iryt;
	int j;
	int left;
	int temp;
	int x;

	if (n <= 0)
	{
		exit(1);
	}

	if (k <= 0)
	{
		exit(1);
	}

	if (n < k)
	{
		exit(1);
	}

	left = 1;
	iryt = n;

	for (; ; )
	{
		if (iryt <= left)
		{
			frac = a[k - 1];
			break;
		}

		x = a[k - 1];
		i = left;
		j = iryt;

		for (;;)
		{
			if (j < i)
			{
				if (j < k)
				{
					left = i;
				}
				if (k < i)
				{
					iryt = j;
				}
				break;
			}
			while (a[i - 1] < x)
			{
				i = i + 1;
			}
			while (x < a[j - 1])
			{
				j = j - 1;
			}

			if (i <= j)
			{
				temp = a[i - 1];
				a[i - 1] = a[j - 1];
				a[j - 1] = temp;
				i = i + 1;
				j = j - 1;
			}
		}
	}

	return frac;
}

int ImageDenoiser::i4vec_median(int n, int* a)
{
	int k = (n + 1) / 2;
	return i4vec_frac(n, a, k);
}
