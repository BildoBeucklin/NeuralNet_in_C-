#include "einheitskreis.h"
#include "integrate.h"

double integrate (double (*f)(double x), double l, double r, int steps)
{
    double x = 0;
    double  b,A,y;
    b = (r - l)/steps; //breite eines Integralbalkens bzw. der x-wert zum Start
#pragma omp parallel for reduction(+:A)
    for(int i = 0; i<steps; i++)
    {
        x = b * i;
        y = f(x);
        A = A + (b*y);        //Fläche + produkt aus breite und y wert der Funktion
    }

    return A;
}