#ifndef _NEURONMATH_H
#define _NEURONMATH_H

/*Kreuzprodukt*/
float dotProduct(float* a, float* b, int n);
/*Sigmoid Funktion*/
float sigmoid(float x);
/*die ableitung von Sigmoid, wenn sigmoid selbst der input*/
float derivedFunktion(float f);
/*eine aktivierungsfunktion*/
float reLU(float x);

#endif // !_NEURONMATH_H
