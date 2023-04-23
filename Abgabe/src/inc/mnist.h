#ifndef _MNIST_H
#define _MNIST_H

#include "init.h"

static const char train_data_path[] = "./data/mnist_train.csv";
static const char test_data_path[] = "./data/mnist_test.csv";

/*ließt eine csv datei ein und speichert es in einem array*/
void read_csv(int row, const char* filename, double** data);
/*lädt die mnist daten in das array*/
int load_mnist(struct net* net, int row, const char* path, float** label_data, float*** nodes);

#endif
