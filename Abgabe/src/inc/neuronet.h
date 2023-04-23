#ifndef _NEURONET_H
#define _NEURONET_H

#include "init.h"

/*setzt Learn Factors auf 0
ist mit pragma omp parallel*/
void resetLearnFactors(struct net* net, struct stp* stp);	

/*testet, das Neuronale Netz*/
float testNet(struct net* net, struct stp* stp);

/*vorformulierte Funktion zum trainieren des Mnist Datensatzes
ist mit pragma omp parallel simd*/
void trainMnist(struct net* net, struct stp* stp);

/*trainiert jeden input*/
void trainNeuralNet(struct net* net, struct stp* stp);

/* eine funktion um den output des netztes zu errechnen*/
void feedForward(struct net* net, struct stp* stp, int start, int ende, float*** nodes);

/*errechnet die zurueckpropagierten learn faktoren
ist mit pragma omp parallel simd*/
void deltaBackProp(struct net* net, struct stp* stp, int row);

/*errechnet den loss*/
void deltaShould(struct net* net, struct stp* stp, int row);

/*wendet den learn faktor auf die gewichte an
ist mit pragma omp parallel simd*/
void backProp(struct net* net, struct stp* stp, int row);

#endif // !_NEURONET_H
