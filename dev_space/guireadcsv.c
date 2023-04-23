#include "inc/mnist.h"
#include "inc/init.h"
#include "inc/guireadcsv.h"
#include "inc/writecsv.h"

void guireadcsv(struct net* net, struct stp* stp){

    FILE* weights_file;
	weights_file = fopen(weights_path, "r+");
	
	

	int i = 0;
	char line[4098];
	while (fgets(line, 4098, weights_file))
	{
		char* tmp = strdup(line);
		int j = 0;
		const char* tok;
		for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, ",")){
			for (int layer = 0; layer < stp->n_layer - 1; layer++){
		        for (int past_node = 0; past_node < net->n_nodes[layer + 1]; past_node++){
			        for (int node = 0; node < net->n_nodes[layer]; node++){
				        net->weights[layer][past_node][node] = atof(tok);
                        printf("%lf\n", net->weights[layer][past_node][node]);
                        j++;
			        }
		        }
	        }
		}
        i++;

        free(tmp);
    }
    fclose(weights_file);
    
    FILE* biases_file;
	biases_file = fopen(biases_path, "r+");

    while (fgets(line, 4098, biases_file))
	{
		char* tmp = strdup(line);
		int j = 0;
		const char* tok;
		for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, ",")){
			for (int layer = 0; layer < stp->n_layer - 1; layer++){
		        for (int past_node = 0; past_node < net->n_nodes[layer + 1]; past_node++){
			        net->biases[layer][past_node] = atof(tok);
                    printf("%lf\n", net->biases[layer][past_node]);
		        }
	        }
		}
		i++;

		free(tmp);
    }
	
	fclose(biases_file);

	printf("readcsv done!\n");
}

