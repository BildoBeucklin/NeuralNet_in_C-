#include "src/inc/neuronet.h"

/*
		agrs 1. epochs, 2. learn rate, 3. batch size, 4. n test, 5. n train, 6. layer, 7. inputs, 8. hidden nodes, 9. outputs
		./net-exe 10 0.01 20 10000 60000 5 784 50 10
*/
int main(int argc, char* argv[])
{
	if (argc != 10) {
		exit(1);
	}
	struct net* net;
	struct stp* stp;
	net = (struct net*)malloc(sizeof(struct net));
	stp = (struct stp*)malloc(sizeof(struct stp));
	stp->epochs = strtol(argv[1], NULL, 10);
	stp->learn_rate = atof(argv[2]);
	stp->batch_size = atof(argv[3]);
	stp->n_test = atof(argv[4]);
	stp->n_train = atof(argv[5]);
	stp->n_layer = atof(argv[6]);
	stp->n_inputs = atof(argv[7]);
	stp->n_hidden_nodes = atof(argv[8]);
	stp->n_outputs = atof(argv[9]);
	initNet(net, stp);
	trainMnist(net, stp);
	//save weights as csv
	free(net);
	free(stp);
	printf("finish");
	return 0;
}
