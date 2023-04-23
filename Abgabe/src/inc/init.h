#ifndef _INIT_H
#define _INIT_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/*beinhaltet setup variabeln*/
typedef struct stp
{
	int batch_size;
	int n_train;
	int n_test;
	int epochs;
	float learn_rate;
	int n_layer;
	int n_outputs;
	int n_inputs;
	int n_hidden_nodes;
	//int act_func;	
} setup;

/*beinhaltet die netzwerk varibaeln*/
typedef struct net
{
	double** mnist_data;			//die von mnist_extrahierten daten
	float** train_label_data;		//train soll werte
	float** test_label_data;		//test soll werte
	float** biases;					//biases[layer][past_node]
	/*
		weights[layer in dem wir uns befinden][knoten des darauffolgenden layer][die knoten in diesem layer]
		und immer von vorne nach hinten, also weights[0] beinhaltet die flanken die auf den input zeigen und in
		der letzten Schich sind die Flanken auf die der Output zeigt
	*/
	float*** weights;
	float** learn_factors;			//alle learn_Factors sortiert nach nodes
	float*** nodes; 				//alle Neruronen
	float*** test_nodes;			//alle Neuronen für Test
	int* n_nodes; 					//anzahl der Neuronen in jeder schicht von input bis output

} network;

/*initslisiert das netz muss immer zu erst aufgerufen werden*/
void initNet(struct net* net, struct stp* stp);

/*extrahiert label aus einem array, wenn diese am Anfang stehen*/
void extractLabel(float** data_arr, double** arr, int n_input);

/*extrahiert den input aus einem array, wenn am Anfang das label steht*/
void extractImages(float*** data_arr, double** arr, int n_input, int n_input_units);

/*deinitalisiert das netzt*/
void deInitNet(struct net* net, struct stp* stp);

#endif // _INIT_H
