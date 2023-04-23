#include "inc/neuronet.h"
#include "inc/mnist.h"
#include "inc/netPy.h"

struct stp * stpInit(int epochs, float learn_rate, int batch_size, int n_layer, int n_hidden, int n_test, int n_train, int n_inputs, int n_outputs)
{
	struct stp* stp;
	stp = (struct stp*)malloc(sizeof(struct stp));
	stp->epochs = epochs;
	stp->learn_rate = learn_rate;
	stp->batch_size = batch_size;
	stp->n_test = n_test;
	stp->n_train = n_train;
	stp->n_layer = n_layer;
	stp->n_inputs = n_inputs;
	stp->n_hidden_nodes = n_hidden;
	stp->n_outputs = n_outputs;
	return stp;
}

struct net* pyFeedForward(struct stp* stp, struct net* net, int start, int ende, int isTest)
{
	if(isTest==1)
	{
		feedForward(net,stp,start,ende,net->test_nodes);	
	}
	else
	{
			feedForward(net,stp,start,ende,net->nodes);
	}

	return net;
}

struct net* pyTrainMnist(struct stp* stp, struct net* net)
{
	load_mnist(net, stp->n_train, train_data_path, net->train_label_data, net->nodes);
	load_mnist(net, stp->n_test, test_data_path, net->test_label_data, net->test_nodes);
	trainNeuralNet(net, stp);
	return net;
}

struct net* netInit(struct stp* stp)
{
		struct net* net;
		net = (struct net*)malloc(sizeof(struct net));
		initNet(net, stp);
		return net;
}

float MnistNet(int epochs, float learn_rate, int batch_size, int n_layer, int n_hidden)
{
	struct net* net;
	struct stp* stp;
	net = (struct net*)malloc(sizeof(struct net));
	stp = (struct stp*)malloc(sizeof(struct stp));
	stp->epochs = epochs;
	stp->learn_rate = learn_rate;
	stp->batch_size = batch_size;
	stp->n_test = 10000;
	stp->n_train = 60000;
	stp->n_layer = n_layer;
	stp->n_inputs = 784;
	stp->n_hidden_nodes = n_hidden;
	stp->n_outputs = 10;
	initNet(net, stp);
	trainMnist(net, stp);
	float mse = 0.0;
	mse = testNet(net, stp);
	free(net);
	free(stp);

	return mse;
}
