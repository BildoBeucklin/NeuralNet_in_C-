#ifndef _NETPY_H
#define _NETPY_H

float MnistNet(int epochs, float learn_rate, int batch_size, int n_layer, int n_hidden);

struct net* pyTrainMnist(struct stp* stp, struct net* net);

struct net* netInit(struct stp* stp);

struct stp* stpInit(int epochs, float learn_rate, int batch_size, int n_layer, int n_hidden, int n_test, int n_train, int n_inputs, int n_outputs);

struct net* pyFeedForward(struct stp* stp, struct net* net, int start, int ende, int isTest);

#endif // !_NETPY_H
