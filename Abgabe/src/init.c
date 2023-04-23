#include "inc/init.h"


#define N_TRAIN_ROW 60000

void initNet(struct net* net, struct stp* stp)
{
	net->train_label_data = (float**)calloc(stp->n_train, sizeof(float*));
	for (int i = 0; i < stp->n_train; ++i)
	{
		net->train_label_data[i] = (float*)calloc(stp->n_outputs, sizeof(float));
	}

	net->test_label_data = (float**)calloc(stp->n_test, sizeof(float*));
	for (int i = 0; i < stp->n_test; ++i)
	{
		net->test_label_data[i] = (float*)calloc(stp->n_outputs, sizeof(float));
	}

	net->n_nodes = (int*)calloc(stp->n_layer, sizeof(int));
	for (int i = 0; i < stp->n_layer; i++)
	{
		if (i == 0)
		{
			net->n_nodes[i] = stp->n_inputs;
		}
		else if (i > 0 && i < stp->n_layer - 1)
		{
			net->n_nodes[i] = stp->n_hidden_nodes;
		}
		else
		{
			net->n_nodes[i] = stp->n_outputs;
		}
	}

	net->nodes = (float***)calloc(stp->n_train, sizeof(float**));
	for (int i = 0; i < stp->n_train; i++)
	{
		net->nodes[i] = (float**)calloc(stp->n_layer, sizeof(float*));
		for (int j = 0; j < stp->n_layer; ++j)
		{
			net->nodes[i][j] = (float*)calloc(net->n_nodes[j], sizeof(float));
		}
	}

	net->test_nodes = (float***)calloc(stp->n_test, sizeof(float**));
	for (int i = 0; i < stp->n_test; i++)
	{
		net->test_nodes[i] = (float**)calloc(stp->n_layer, sizeof(float*));
		for (int j = 0; j < stp->n_layer; ++j)
		{
			net->test_nodes[i][j] = (float*)calloc(net->n_nodes[j], sizeof(float));
		}
	}

	net->learn_factors = (float**)calloc(stp->n_layer - 1, sizeof(float*));
	for (int i = 0; i < stp->n_layer - 1; i++)
	{
		net->learn_factors[i] = (float*)calloc(net->n_nodes[i + 1], sizeof(float));
	}

	net->biases = (float**)calloc((stp->n_layer - 1), sizeof(float*));
	for (int i = 0; i < stp->n_layer - 1; i++)
	{
		net->biases[i] = (float*)calloc((net->n_nodes[i + 1]), sizeof(float));
		for (int j = 0; j < net->n_nodes[i + 1]; j++)
		{
			net->biases[i][j] = ((float)rand() / (float)RAND_MAX) - ((float)rand() / (float)RAND_MAX);
		}
	}

	net->weights = (float***)calloc((stp->n_layer - 1), sizeof(float**));
	for (int i = 0; i < stp->n_layer - 1; i++)
	{
		net->weights[i] = (float**)calloc((net->n_nodes[i + 1]), sizeof(float*));
		for (int j = 0; j < net->n_nodes[i + 1]; j++)
		{
			net->weights[i][j] = (float*)calloc((net->n_nodes[i]), sizeof(float));
			for (int k = 0; k < net->n_nodes[i]; k++)
			{
				net->weights[i][j][k] = ((float)rand() / (float)RAND_MAX) - ((float)rand() / (float)RAND_MAX);
			}
		}
	}

	net->mnist_data = (double**)malloc(N_TRAIN_ROW * sizeof(double*));
	for (int i = 0; i < N_TRAIN_ROW; ++i) {
		net->mnist_data[i] = (double*)malloc(net->n_nodes[0] * sizeof(double));
	}
}

void extractLabel(float** data_arr, double** arr, int n_input)
{
	int temp;
	int j = 0;
#pragma omp parallel for simd
	for (int i = 0; i < n_input; i++)
	{
		temp = arr[i][j];
		data_arr[i][temp] = 1;
	}
}

void extractImages(float*** data_arr, double** arr, int rows, int n_input_units)
{
#pragma omp parallel for simd
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < n_input_units; j++)
		{
			data_arr[i][0][j] = (float)(arr[i][j] / 255);			//zwischen 0 ... 1 
		}
	}
}
//funktioniet noch nicht
void deInitNet(struct net* net, struct stp* stp)
{
	float* current_sing_ptr;
	float** current_doub_ptr;

	for (int j = 0; j < (net->n_nodes[stp->n_layer - 2]); j++)
	{
		current_sing_ptr = net->weights[stp->n_layer - 2][j];
		free(current_sing_ptr);
	}
	free(net->weights[stp->n_layer - 2]);
	for (int i = 0; i < stp->n_layer; i++)
	{
		for (int j = 0; j < net->n_nodes[i + 1]; j++)
		{
			current_sing_ptr = net->weights[i][j];
			free(current_sing_ptr);
		}
		current_doub_ptr = net->weights[i];
		free(current_doub_ptr);
	}
	free(net->weights);

	for (int i = 1; i < stp->n_layer; i++)
	{
		current_sing_ptr = net->learn_factors[i - 1];
		free(current_sing_ptr);
	}
	free(net->learn_factors);

	for (int i = 0; i < stp->n_test; i++)
	{
		for (int j = 0; j < stp->n_layer; j++)
		{
			current_sing_ptr = net->nodes[i][j];
			free(current_sing_ptr);
		}
		current_doub_ptr = net->nodes[i];
		free(current_doub_ptr);
	}
	free(net->nodes);

	free(net->n_nodes);

	for (int i = 1; i < stp->n_test; i++)
	{
		current_sing_ptr = net->test_label_data[i];
		free(current_sing_ptr);
	}
	free(net->test_label_data);

	for (int i = 1; i < stp->n_train; i++)
	{
		current_sing_ptr = net->train_label_data[i];
		free(current_sing_ptr);
	}
	free(net->train_label_data);

}