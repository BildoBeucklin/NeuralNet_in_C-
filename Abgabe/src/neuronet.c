#include "inc/neuronet.h"
#include "inc/neuronmath.h"
#include "inc/init.h"
#include "inc/mnist.h"

void feedForward(struct net* net, struct stp* stp, int start, int ende, float*** nodes)
{
#pragma omp parallel for simd
	for (int row = start; row < ende; row++)
	{
		for (int layer = 0; layer < stp->n_layer - 1; layer++)
		{
			for (int node = 0; node < net->n_nodes[layer + 1]; node++)
			{
				nodes[row][layer + 1][node] = sigmoid(dotProduct(nodes[row][layer], net->weights[layer][node], net->n_nodes[layer]) + net->biases[layer][node]);
			}
		}
	}
}

void deltaBackProp(struct net* net, struct stp* stp, int row)
{
	for (int layer = stp->n_layer - 2; layer > 0; layer--)
	{
		for (int node = 0; node < net->n_nodes[layer]; node++)
		{
			for (int past_node = 0; past_node < net->n_nodes[layer + 1]; past_node++)
			{
				net->learn_factors[layer - 1][node] += net->learn_factors[layer][past_node] * net->weights[layer][past_node][node];
			}
			net->learn_factors[layer - 1][node] *= derivedFunktion(net->nodes[row][layer][node]);
		}
	}

}

void deltaShould(struct net* net, struct stp* stp, int row)
{
	int out_layer = stp->n_layer - 1;
	for (int node = 0; node < net->n_nodes[out_layer]; node++)
	{
		net->learn_factors[out_layer - 1][node] = derivedFunktion(net->nodes[row][out_layer][node]) * -2 * (net->nodes[row][out_layer][node] - net->train_label_data[row][node]);
	}

}

void backProp(struct net* net, struct stp* stp, int row)
{
	resetLearnFactors(net, stp);
	deltaShould(net, stp, row);
	deltaBackProp(net, stp, row);
#pragma omp parallel for simd
	for (int layer = 0; layer < stp->n_layer - 1; layer++)
	{
		for (int past_node = 0; past_node < net->n_nodes[layer + 1]; past_node++)
		{
			for (int node = 0; node < net->n_nodes[layer]; node++)
			{
				net->weights[layer][past_node][node] += net->nodes[row][layer][node] * net->learn_factors[layer][past_node] * stp->learn_rate;
			}
			net->biases[layer][past_node] += net->learn_factors[layer][past_node] * stp->learn_rate;
		}
	}
}

void resetLearnFactors(struct net* net, struct stp* stp)
{
#pragma omp parallel for
	for (int i = 0; i < stp->n_layer - 1; i++)
	{
		for (int j = 0; j < net->n_nodes[i + 1]; j++)
		{
			net->learn_factors[i][j] = 0;
		}
	}
}

float testNet(struct net* net, struct stp* stp)
{
	int* predictions;
	predictions = (int*)calloc(stp->n_test, sizeof(int));
	int* labels;
	labels = (int*)calloc(stp->n_test, sizeof(int));
	float mse = 0;
	int prediction;
	printf("vor feed");
	feedForward(net, stp, 0, stp->n_test, net->test_nodes);
	printf("nach feed\n");
#pragma omp parallel for simd
	for (int test_row = 0; test_row < stp->n_test; test_row++)
	{
		prediction = 0;
		for (int i = 0; i < stp->n_outputs; i++)
		{
			if (net->test_nodes[test_row][stp->n_layer - 1][prediction] <= net->test_nodes[test_row][stp->n_layer - 1][i])
			{
				prediction = i;
			}
			if (net->test_label_data[test_row][i] == 1)
			{
				labels[test_row] = i;
			}
		}
		predictions[test_row] = prediction;
	}
#pragma omp parallel for simd
	for (int j = 0; j < stp->n_test; j++)
	{
		if (predictions[j] == labels[j])
		{
#pragma omp atomic
			mse++;
		}
	}
	mse = (mse / stp->n_test) * 100;
	printf("vor free");
	free(predictions);
	free(labels);
	printf("nach free");
	return mse;
}

void trainMnist(struct net* net, struct stp* stp)
{
	load_mnist(net, stp->n_train, train_data_path, net->train_label_data, net->nodes);
	load_mnist(net, stp->n_test, test_data_path, net->test_label_data, net->test_nodes);
	for (int i = 0; i < stp->epochs; i++)
	{
		int row = 0;
		while (row < stp->n_train)
		{
			feedForward(net, stp, row, row + stp->batch_size, net->nodes);
			for (int j = 0; j < stp->batch_size; j++)
			{
				backProp(net, stp, row + j);
			}
			row += stp->batch_size;
		}
		stp->learn_rate *= 0.999;
	}
}

void trainNeuralNet(struct net* net, struct stp* stp)
{
	for (int i = 0; i < stp->epochs; i++)
	{
		int row = 0;
		while (row < stp->n_train)
		{
			feedForward(net, stp, row, row + stp->batch_size, net->nodes);
			for (int j = 0; j < stp->batch_size; j++)
			{
				backProp(net, stp, row + j);
			}
			row += stp->batch_size;
		}
		stp->learn_rate *= 0.999;
	}
}
