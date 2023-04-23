#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include "integrate.h"
#include "einheitskreis.h"

int main(int argc, char *argv[])
{
    char *p;
    int input;
    double A,pi;
    long temp = strtol(argv[1], &p, 10);
    input = temp;
    A = integrate(einheitskreis,0,1,input);
    pi = A*4;
    printf("%.50lf\n",pi);
    return 0;
}