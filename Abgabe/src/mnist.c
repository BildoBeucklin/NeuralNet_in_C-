#include "inc/mnist.h"
#include "inc/init.h"


int load_mnist(struct net* net, int row, const char* path, float** label_data, float*** nodes)
{
	read_csv(row, path, net->mnist_data);
	extractLabel(label_data, net->mnist_data, row);
	extractImages(nodes, net->mnist_data, row, net->n_nodes[0]);
	return 0;
}

void read_csv(int row, const char* filename, double** data) {
	FILE* file;

	file = fopen(filename, "r");
	int i = 0;
	char line[4098];
	while (fgets(line, 4098, file) && (i < row))
	{
		char* tmp = strdup(line);
		int j = 0;
		const char* tok;
		for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, ","))
		{
			data[i][j] = atof(tok);
		}
		i++;

		free(tmp);
	}
	fclose(file);
}