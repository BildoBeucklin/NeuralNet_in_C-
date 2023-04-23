#include "inc/neuronmath.h"
#include <math.h>

float dotProduct(float* a, float* b, int n)
{
	float sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += a[i] * b[i];
	}
	return sum;
}

float sigmoid(float x)
{
	float f;
	f = 1 / (1 + exp(-x));
	return f;
}

float derivedFunktion(float f) 		//mit sigmoid als input
{
	;
	float df = f * (1 - f);
	return df;
}

float reLU(float x)
{
	if (x > 0)
	{
		return x;
	}
	else
	{
		return 0;
	}
}