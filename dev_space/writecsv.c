#include "inc/mnist.h"
#include "inc/init.h"
#include "inc/writecsv.h"
#include "inc/init.h"


void writecsv(struct net* net, struct stp* stp){
	FILE* weights_file;
	weights_file = fopen(weights_path, "w+");
	
	FILE* biases_file;
	biases_file = fopen(biases_path, "w+");

	for (int layer = 0; layer < stp->n_layer - 1; layer++)
	{
		for (int past_node = 0; past_node < net->n_nodes[layer + 1]; past_node++)
		{
			for (int node = 0; node < net->n_nodes[layer]; node++)
			{
				fprintf(weights_file, "%lf,", net->weights[layer][past_node][node]);
				
			}
			fprintf(biases_file, "%lf,", net->biases[layer][past_node]);
		}
	}

	fclose(weights_file);
	fclose(biases_file);

	printf("Writecsv done!\n");
}
