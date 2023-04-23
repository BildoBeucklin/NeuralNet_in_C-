#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

//for windows wegen unistd.h
#ifdef WIN32
#include <io.h>
#define F_OK 0
#define access _access
#endif

#define EULER_NUMBER_F 2.71828182846

//fuer csv datei
#define N_TRAIN_ROW 60000				//gerade nur 10000 wieder auf 60000 setzen
#define N_TEST_ROW 10000
#define N_INPUT_UNITS 784				//28*28 Anzahl der Pixel
#define N_HIDDEN_NODES	1000
#define OFFSET_IMAGE 1
#define N_OUTPUTS 10 

static const char train_data_path[] = "./data/mnist_train.csv";
static const char test_data_path[] = "./data/mnist_test.csv";

typedef struct stp
{
	int n_train;
	int n_test;
	int epochs;
	float learn_rate;

	int n_layer;
	int n_outputs;
	int n_inputs;

	int n_hidden_nodes;

	//int act_func;
	//int dataset;
	
} setup;


typedef struct net
{
	double** mnist_data;

	//float** train_image_data;
	float** train_label_data;
	//float** test_image_data;
	float** test_label_data;

	float** biases;
	/*float** outputs;*/
	/*float** test_outputs;*/

	/*
		weights[layer in dem wir uns befinden][knoten des darauffolgenden layer][die knoten in diesem layer]
		und immer von vorne nach hinten, also weights[0] beinhaltet die flanken die auf den input zeigen und in
		der letzten Schich sind die Flanken auf die der Output zeigt
	*/
	float*** weights; 
	float*** delta_weights;

	float** learn_factors;

	//float*** nodes_input;			//vor sigmoid
	float*** nodes; 				//alle Neruronen
	float*** test_nodes;
	int* n_nodes; 				//anzahl der Neuronen in jeder schicht von input bis output

} network;


void resetLearnFactors(struct net* net, struct stp* stp);
void testLoadData(struct net* net, struct stp* stp);
void testNeuralNet(struct net* net, struct stp* stp);
void initNet(struct net* net, struct stp* stp);
void deInitNet(struct net* net, struct stp* stp);
float dotProduct(float* a, float* b, int n);
float sigmoid(float x);
float derivedFunktion(float f);
void extractLabel(float** data_arr, double** arr, int n_input);
void extractImages(float*** data_arr, double** arr, int n_input, int n_input_units);
void read_csv(int row, /*int col,*/ const char* filename, double** data);
int load_mnist(struct net* net, int row, const char* path, float** label_data, float*** nodes);
void trainNeuralNet(struct net* net, struct stp* stp);
//void deltaLearn(struct net* net, struct stp* stp, int output_layer);
void waitFor(unsigned int secs);
float reLU(float x);
void feedForward(struct net* net, struct stp* stp, int x, float*** nodes);
void resetLearnFactors(struct net* net, struct stp* stp);
void deltaBackProp(struct net* net, struct stp* stp, int row);
void deltaShould(struct net* net, struct stp* stp, int row);
void backProp(struct net* net, struct stp* stp, int row);
//void backProp(struct net* net, struct stp* stp);
void learnFactor(struct net* net, struct stp* stp, int row);
void reCallocNodesRows(struct stp* stp, struct net* net, int n_old, int n_new);

void initNet(struct net* net, struct stp* stp)
{
	//net->test_outputs = (float**)calloc(stp->n_train, sizeof(float*));
	//for (int i = 0; i < stp->n_train; ++i)
	//{
	//	net->test_outputs[i] = (float*)calloc(stp->n_outputs, sizeof(float));
	//}

	net->train_label_data = (float**)calloc(stp->n_train, sizeof(float*));
	for (int i = 0; i < stp->n_train; ++i)
	{
		net->train_label_data[i] = (float*)calloc(stp->n_outputs, sizeof(float));
	}

	//net->train_image_data = (float**)calloc(stp->n_train, sizeof(float*));
	//for (int i = 0; i < stp->n_train; ++i)
	//{
	//	net->train_image_data[i] = (float*)calloc(stp->n_inputs, sizeof(float));
	//}

	net->test_label_data = (float**)calloc(stp->n_test, sizeof(float*));
	for (int i = 0; i < stp->n_test; ++i)
	{
		net->test_label_data[i] = (float*)calloc(stp->n_outputs, sizeof(float));
	}

	//net->test_image_data = (float**)calloc(stp->n_test, sizeof(float*));
	//for (int i = 0; i < stp->n_test; ++i)
	//{
	//	net->test_image_data[i] = (float*)calloc(stp->n_inputs, sizeof(float));
	//}


	net->n_nodes = (int*)calloc(stp->n_layer, sizeof(int));		
	for (int i = 0; i < stp->n_layer; i++)//immer plus 1 für bias node hat immer den wert 1 außer output !!!erstmal geändert ohne bias
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
		printf("\nn_nodes[%d] = %d", i, net->n_nodes[i]);
	}

	//net->nodes = (float***)calloc(stp->n_train, sizeof(float**));
	//for (int i = 0; i < stp->n_train; i++)
	//{
	//	net->nodes[i] = (float**)calloc(stp->n_layer, sizeof(float*));
	//	for (int j = 0; j < stp->n_layer; ++j)
	//	{
	//		if (j == stp->n_layer - 1)
	//		{
	//			net->nodes[i][j] = (float*)calloc(net->n_nodes[j], sizeof(float));
	//		}
	//		else
	//		{
	//			net->nodes[i][j] = (float*)calloc(net->n_nodes[j], sizeof(float));
	//			net->nodes[i][j][0] = 1;			//der bias
	//		}
	//	}
	//}

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
	//net->test_nodes = (float***)calloc(stp->n_test, sizeof(float**));
	//for (int i = 0; i < stp->n_test; i++)
	//{
	//	net->test_nodes[i] = (float**)calloc(stp->n_layer, sizeof(float*));
	//	for (int j = 0; j < stp->n_layer; ++j)
	//	{
	//		net->test_nodes[i][j] = (float*)calloc(net->n_nodes[j], sizeof(float));
	//		net->test_nodes[i][j][0] = 1;			//der bias 
	//	}
	//}

	//net->nodes_input = (float***)calloc(stp->n_train, sizeof(float**));
	//for (int i = 0; i < stp->n_train; i++)
	//{
	//	net->nodes_input[i] = (float**)calloc(stp->n_layer, sizeof(float*));
	//	for (int j = 1; j < stp->n_layer; ++j)
	//	{
	//		net->nodes_input[i][j-1] = (float*)calloc(net->n_nodes[j], sizeof(float));
	//	}
	//}

	//net->learn_factors = (float**)calloc(stp->n_layer-1, sizeof(float*));
	//for (int i = 1; i < stp->n_layer; i++)
	//{
	//	if (i == stp->n_layer-1)
	//	{
	//		net->learn_factors[i - 1] = (float*)calloc(net->n_nodes[i] , sizeof(float));
	//	}
	//	else
	//	{
	//		net->learn_factors[i - 1] = (float*)calloc(net->n_nodes[i] - 1, sizeof(float));
	//	}
	//}

	net->learn_factors = (float**)calloc(stp->n_layer-1, sizeof(float*));
	for (int i = 0; i < stp->n_layer-1; i++)
	{
		net->learn_factors[i] = (float*)calloc(net->n_nodes[i+1], sizeof(float));
		//printf("test\n");
	}

	net->weights = (float***)calloc((stp->n_layer-1), sizeof(float**));
	for (int i = 0; i < stp->n_layer - 1; i++)
	{
		net->weights[i] = (float**)calloc((net->n_nodes[i + 1]), sizeof(float*));
		for (int j = 0; j < net->n_nodes[i + 1]; j++)
		{
			net->weights[i][j] = (float*)calloc((net->n_nodes[i]), sizeof(float));
			for (int k = 0; k < net->n_nodes[i]; k++)
			{
				net->weights[i][j][k] = ((float)rand() / (float)RAND_MAX) /*- ((float)rand() / (float)RAND_MAX)*/;
			}
		}
	}

	net->delta_weights = (float***)calloc((stp->n_layer - 1), sizeof(float**));
	for (int i = 0; i < stp->n_layer - 1; i++)
	{
		net->delta_weights[i] = (float**)calloc((net->n_nodes[i + 1]), sizeof(float*));
		for (int j = 0; j < net->n_nodes[i + 1]; j++)
		{
			net->delta_weights[i][j] = (float*)calloc((net->n_nodes[i]), sizeof(float));
			for (int k = 0; k < net->n_nodes[i]; k++)
			{
				net->delta_weights[i][j][k] = ((float)rand() / (float)RAND_MAX) /*- ((float)rand() / (float)RAND_MAX)*/;
			}
		}
	}

	//net->weights = (float***)calloc((stp->n_layer), sizeof(float**));
	//for (int i = 0; i < stp->n_layer - 2; i++)
	//{
	//	net->weights[i] = (float**)calloc((net->n_nodes[i + 1]), sizeof(float*));
	//	for (int j = 0; j < (net->n_nodes[i + 1]-1); j++)	//auf den bias zeigt nimand
	//	{
	//		net->weights[i][j] = (float*)calloc((net->n_nodes[i]), sizeof(float));
	//		for (int k = 0; k < net->n_nodes[i]; k++)
	//		{
	//			net->weights[i][j][k] = ((float)rand() / (float)RAND_MAX) - ((float)rand() / (float)RAND_MAX);
	//		}
	//	}
	//}
	////für output layer
	//net->weights[stp->n_layer - 2] = (float**)calloc((net->n_nodes[stp->n_layer-2]), sizeof(float*));
	//for (int j = 0; j < (net->n_nodes[stp->n_layer - 2]); j++)
	//{
	//	net->weights[stp->n_layer - 2][j] = (float*)calloc((net->n_nodes[stp->n_layer - 3]), sizeof(float));
	//	for (int k = 0; k < net->n_nodes[stp-> n_layer - 3]; k++)
	//	{
	//		net->weights[stp->n_layer - 2][j][k] = ((float)rand() / (float)RAND_MAX)- ((float)rand() / (float)RAND_MAX);
	//	}
	//}

	net->mnist_data = (double**)malloc(N_TRAIN_ROW * sizeof(double*));
	for (int i = 0; i < N_TRAIN_ROW; ++i) {
		net->mnist_data[i] = (double*)malloc(net->n_nodes[0] * sizeof(double));
	}
	//net->biases = (float**)malloc((stp->n_layer - 1)* sizeof(float*));
	//for (int i = 1; i < stp->n_layer - 1; i++)
	//{
	//	net->biases[i] = (float*)malloc((net->n_nodes[i]) * sizeof(float));
	//	for (int k = 0; k < net->n_nodes[i]; k++)
	//	{
	//		net->biases[i][k] = ((float)rand() / (float)RAND_MAX) - ((float)rand() / (float)RAND_MAX);
	//	}
	//}
	/*if (stp->n_layer == 2)
	{
		net->weights = (float***)calloc(stp->n_layer-1, sizeof(float**))
		net->weights[i] = (float**)calloc(stp->n_outputs, sizeof(float*));
		for (int j = 0; j < stp->n_outputs; j++)
		{
			net->weights[0][j] = (float*)calloc(stp->n_inputs, sizeof(float));
		}
	}
	else
	{
		net->weights = (float***)calloc(stp->n_layer, sizeof(float**));
		for (int i = 0; i < stp->n_layer; ++i)
		{
			if (i == 0)
			{
				net->weights[i] = (float**)calloc(stp->n_hidden_nodes, sizeof(float*));
				for (int j = 0; j < stp->n_hidden_nodes; j++)
				{
					net->weights[i][j] = (float*)calloc(stp->n_inputs, sizeof(float));
				}
			}
			else if (i > 0 && i < stp->n_layer - 1)
			{
				net->weights[i] = (float**)calloc(stp->n_hidden_nodes, sizeof(float*));
				for (int j = 0; j < stp->n_hidden_nodes; j++)
				{
					net->weights[i][j] = (float*)calloc(stp->n_hidden_nodes, sizeof(float));
				}
			}
			else
			{
				net->weights[i] = (float**)calloc(stp->n_outputs, sizeof(float*));
				for (int j = 0; j < stp->n_outputs; j++)
				{
					net->weights[i][j] = (float*)calloc(stp->n_hidden_nodes, sizeof(float));
				}
			}
		}

		for (int i = 0; i < stp->n_layer; i++)
		{
			if (i == 0)
			{
				for (int j = 0; j < stp->n_hidden_nodes; j++)
				{
					for (int k = 0; k < stp->n_inputs; k++)
					{
						net->weights[i][j][k] = ((float)rand() / (float)RAND_MAX) - ((float)rand() / (float)RAND_MAX);
					}
				}
			}
			else if (i > 0 && i < stp->n_layer - 1)
			{
				for (int j = 0; j < stp->n_hidden_nodes; j++)
				{
					for (int k = 0; k < stp->n_hidden_nodes; k++)
					{
						net->weights[i][j][k] = ((float)rand() / (float)RAND_MAX) - ((float)rand() / (float)RAND_MAX);
					}
				}
			}
			else
			{
				for (int j = 0; j < stp->n_outputs; j++)
				{
					for (int k = 0; k < stp->n_hidden_nodes; k++)
					{
						net->weights[i][j][k] = ((float)rand() / (float)RAND_MAX) - ((float)rand() / (float)RAND_MAX);
					}
				}
			}
		}
	}*/

	printf("\n");
}

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

void waitFor(unsigned int secs) {
	unsigned int retTime = time(0) + secs;   // Get finishing time.
	while (time(0) < retTime);               // Loop until it arrives.
}

float dotProduct(float* a, float* b, int n)
{
	float sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += a[i] * b[i];
	}
	return sum;
}

float sigmoid(float x)
{
	float f;
	f = 1 / (1 + exp(-x));
	return f;
}

float derivedFunktion(float f) 		//mit sigmoid als input
{;
	float df = f * (1 - f);
	return df;
}

float reLU(float x)
{
	if (x > 0)
	{
		return x;
	}
	else
	{
		return 0;
	}
}

//void feedForward(struct net* net, struct stp *stp, int x, float*** nodes)
//{	
//	#pragma omp parallel for
//	for (int row = 0; row < x; row++)
//	{
//		//printf("\n %d", row);
//		for (int layer = 0; layer < stp->n_layer-2; layer++)
//		{
//			for (int node = 1; node < net->n_nodes[layer+1]; node++)			//weil wir nicht auf einen bias forward steppen (den überspringen wir mit plus 1)
//			{
//				nodes[row][layer + 1][node] = sigmoid(dotProduct(nodes[row][layer], net->weights[layer][node-1], net->n_nodes[layer])); 
//			}
//		}
//		
//		for (int node = 0; node < net->n_nodes[stp->n_layer - 1]; node++)  //für letzte schicht, da dort kein Bias
//		{
//			nodes[row][stp->n_layer - 1][node] = dotProduct(nodes[row][stp->n_layer - 2], net->weights[stp->n_layer - 2][node], net->n_nodes[stp->n_layer - 2]);
//			nodes[row][stp->n_layer-1][node] = sigmoid(nodes[row][stp->n_layer - 1][node]);
//		}
//	}
//}
void feedForward(struct net* net, struct stp* stp, int x, float*** nodes)
{
#pragma omp parallel for simd
	for (int row = 0; row < x; row++)
	{
		for (int layer = 0; layer < stp->n_layer - 1; layer++)
		{
			//printf("%d", layer);
			for (int node = 0; node < net->n_nodes[layer + 1]; node++)
			{
				nodes[row][layer + 1][node] = sigmoid(dotProduct(nodes[row][layer], net->weights[layer][node], net->n_nodes[layer]));
			}
		}
	}
}


/*
void deltaLearn(struct net* net, struct stp* stp, int output_layer)		//für die output layer
{
	for (int train_row = 0; train_row < N_TRAIN_ROW; train_row++)
	{
		feedForward(net, net->outputs, net->train_image_data, train_row, N_OUTPUTS);
		for (int i_output = 0; i_output < N_OUTPUTS; i_output++)
		{
			float learnFactor = net->learn_rate * (net->train_label_data[train_row][i_output] - net->outputs[train_row][i_output]);
			for (int i_input = 0; i_input < N_INPUT_UNITS; i_input++)		//real delta learn
			{
				net->weights[i_output][i_input] += net->train_image_data[train_row][i_input] * learnFactor;
			}
		}
	}
}
*/

void learnFactor(struct net* net, struct stp* stp, int row)
{
	for (int layer = stp->n_layer - 2; layer > 0; layer--) // 1. Hidden schicht (vom output geschaut)
	{
		//printf("\nwar learn layer: %d", layer);
		for (int node = 1; node < net->n_nodes[layer]; node++)
		{
			//printf("\nwar learn: %d", node);
			for (int past_node = 0; past_node < net->n_nodes[layer+1]; past_node++)
			{
				//printf("\nwar learn: %d", node);
				net->learn_factors[layer-1][node-1] += net->learn_factors[layer][past_node] * net->weights[layer][past_node][node];
				//printf("\n %f * %f", net->learn_factors[layer][past_node], net->weights[layer][past_node][node]);
			}
			//printf("\nwar learn: %d", node);
			//printf("\nNetz_output: %f", net->nodes[row][layer][node]);
			//printf("\nNetz_output: %f * %f", derivedFunktion(net->nodes[row][layer][node]), net->learn_factors[layer - 1][node - 1]);
			net->learn_factors[layer - 1][node-1] *= derivedFunktion(net->nodes[row][layer][node]) * stp->learn_rate;
			//printf(" f' : %f\n", derivedFunktion(net->nodes[row][layer][node]));
			//printf("LearnFactor[%d][%d]: %f", layer - 1, node, net->learn_factors[layer - 1][node-1]);
		}
	}
}

void deltaBackProp(struct net* net, struct stp* stp,int row)
{
	for (int layer = stp->n_layer - 2; layer > 0; layer--)
	{
		for (int node = 0; node < net->n_nodes[layer]; node++)
		{
			for (int past_node = 0; past_node < net->n_nodes[layer + 1]; past_node++)
			{
				net->learn_factors[layer-1][node] += net->learn_factors[layer][past_node] * net->weights[layer][past_node][node];
			}
			net->learn_factors[layer - 1][node] *= derivedFunktion(net->nodes[row][layer][node]);
		}
	}

}
//https://www.youtube.com/watch?v=QJoa0JYaX1I

void deltaShould(struct net* net, struct stp* stp,int row)
{
	int out_layer = stp->n_layer - 1;
	for (int node = 0; node < net->n_nodes[out_layer]; node++)
	{
		net->learn_factors[out_layer - 1][node] = derivedFunktion(net->nodes[row][out_layer][node]) * (net->train_label_data[row][node] - net->nodes[row][out_layer][node]) ;
		//printf("% f\n", net->learn_factors[out_layer - 1][node]);
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
		}
		
	}

}

void resetLearnFactors(struct net* net, struct stp* stp)
{
	#pragma omp parallel for
	for (int i = 0; i < stp->n_layer-1; i++)
	{
		for (int j = 0; j < net->n_nodes[i+1]; j++)
		{
			net->learn_factors[i][j] = 0;
		}
	}
}

//void backProp2(struct net* net, struct stp* stp, int row)
//{
//	resetLearnFactors(net, stp);
//	deltaShould(net, stp, row);
//	deltaBackProp(net, stp, row);
//	if (stp->n_hidden_nodes > 0)
//	{
//		//#pragma omp parallel for
//		for (int layer = 0; layer < stp->n_layer - 1; layer++)
//		{
//			if (layer == stp->n_layer - 2)
//			{
//				for (int past_node = 0; past_node < net->n_nodes[layer + 1]; past_node++)
//				{
//					for (int node = 0; node < net->n_nodes[layer]; node++)
//					{
//						net->weights[layer][past_node][node] += net->learn_factors[layer][past_node] * stp->learn_rate  * net->nodes[row][layer][node] * derivedFunktion(net->nodes[row][layer + 1][past_node]);
//					}
//				}
//			}
//			else
//			{
//				for (int past_node = 1; past_node < net->n_nodes[layer + 1]; past_node++)
//				{
//					for (int node = 0; node < net->n_nodes[layer]; node++)
//					{
//						net->weights[layer][past_node - 1][node] += net->learn_factors[layer][past_node - 1] * stp->learn_rate  * net->nodes[row][layer][node] * derivedFunktion(net->nodes[row][layer+1][past_node]);
//					}
//				}
//			}
//
//		}
//	}
//}

//void backProp(struct net* net, struct stp* stp)
//{
//	feedForward(net, stp, stp->n_train);
//	//if (stp->n_layer == 2) 			// dann deltaLearn
//	//{
//	//	for (int train_row = 0; train_row < stp->n_train; train_row++)
//	//	{
//	//		printf("\nrow: %d", net->n_nodes[0]);
//	//		for (int i = 0; i < net->n_nodes[1]; i++)
//	//		{
//	//			//printf("\nZahl %d => pred:  %f| ist: %f", i, net->nodes[train_row][1][i], net->train_label_data[train_row][i]);
//	//			net->learn_factors[stp->n_layer - 1][i] = learnFactor(stp->learn_rate, net->nodes_input[train_row][stp->n_layer - 2][i], net->train_label_data[train_row][i - 1]);
//	//			printf("\n learnfac: %f", net->train_label_data[train_row][i-1] - net->nodes[train_row][1][i]);
//	//			for (int j = 0; j < net->n_nodes[0]; j++)
//	//			{
//	//				//printf("\nwar da: %f", learn_factor);
//	//				net->weights[0][i][j] += net->nodes_input[train_row][0][j] * net->learn_factors[stp->n_layer - 1][i];
//	//				//printf("\nweights[0][%d][%d] = %f", i, j, net->weights[0][i][j]);
//	//			}
//	//			//printf("\nZahl %d => pred:  %f| ist: %f", i, net->nodes[train_row][stp->n_layer - 1][i], net->train_label_data[train_row][i]);
//	//		}
//	//	}
//	//}
//	//else
//	//{
//		int out_layer = stp->n_layer - 1;
//		for(int train_row = 0; train_row < stp->n_train; train_row++)
//		{ 
//			//printf("\rROW: %d", train_row);
//			//printf("\nwar da: %f", stp->learn_rate);
//			for (int node = 0; node < net->n_nodes[out_layer]; node++)
//			{
//
//				net->learn_factors[out_layer - 1][node] = /*stp->learn_rate * derivedFunktion(net->nodes[train_row][out_layer][node]) **/ (net->train_label_data[train_row][node] - net->nodes[train_row][out_layer][node]);
//			}
//			learnFactor(net, stp, train_row);
//			for(int layer = stp->n_layer-1; layer > 0; layer --)
//			{
//				if(layer == stp->n_layer-1)
//				{
//					for (int node = 0; node < net->n_nodes[layer]; node++)
//					{
//						//printf("\nLearnFactor[%d][%d]: %f", layer - 1, node-1, net->learn_factors[layer - 1][node-1]);
//						//printf("\rrow: %d", train_row);
//						for (int pre_node = 0; pre_node < net->n_nodes[layer - 1]; pre_node++)
//						{
//							//printf("Layer= %d | Node= %d | Pre Node= %d\n", layer, node, pre_node);
//							//printf("Gewichtsanpassung: %f + %f * %f\n ", net->weights[layer - 1][node][pre_node], net->learn_factors[layer - 1][node], net->nodes[train_row][layer - 1][pre_node]);
//							net->weights[layer - 1][node][pre_node] += net->nodes[train_row][layer - 1][pre_node] * net->learn_factors[layer - 1][node];
//							//printf(" %f \n", (net->nodes[train_row][layer - 1][pre_node] * net->learn_factors[layer - 1][node]));
//							//printf("\nROW: %d layer: %d node: %d  pre_node : %d = %f", train_row, layer-1,node-1, pre_node, net->weights[layer - 1][node - 1][pre_node]);
//							//printf("Gewicht = %f\n ", net->weights[layer - 1][node][pre_node]);
//						}
//						//printf("LearnFactor[%d][%d]: %f\n", layer - 1, node-1, net->learn_factors[layer - 1][node-1]);
//					}
//				}
//				else
//				{
//					for (int node = 1; node < net->n_nodes[layer]; node++)
//					{
//						//printf("\nLearnFactor[%d][%d]: %f", layer - 1, node-1, net->learn_factors[layer - 1][node-1]);
//						//printf("\rrow: %d", train_row);
//						for (int pre_node = 0; pre_node < net->n_nodes[layer - 1]; pre_node++)
//						{
//							//printf("Layer= %d | Node= %d | Pre Node= %d\n", layer, node, pre_node);
//							//printf("Gewichtsanpassung: %f + %f * %f\n ", net->weights[layer - 1][node-1][pre_node], net->learn_factors[layer - 1][node-1], net->nodes[train_row][layer - 1][pre_node]);
//							net->weights[layer - 1][node-1][pre_node] += net->nodes[train_row][layer - 1][pre_node] * net->learn_factors[layer-1][node-1];
//							//printf(" %f \n", (net->nodes[train_row][layer - 1][pre_node] * net->learn_factors[layer - 1][node-1]));
//							//printf("\nROW: %d layer: %d node: %d  pre_node : %d = %f", train_row, layer-1,node-1, pre_node, net->weights[layer - 1][node - 1][pre_node]);
//							//printf("Gewicht = %f\n ", net->weights[layer - 1][node][pre_node]);
//						}
//						//printf("LearnFactor[%d][%d]: %f\n", layer - 1, node-1, net->learn_factors[layer - 1][node-1]);
//					}
//				}
//			}
//		}
//	//	
//	//}
//	//printf("\nROW:");
//	//waitFor(3);
//	//printf("\nROW:");
//	//resetLearnFactors(net, stp);
//}

void extractLabel(float** data_arr, double** arr, int n_input)
{
	int temp;
	int j = 0;
	#pragma omp parallel for
	for (int i = 0; i < n_input; i++)
	{
		temp = arr[i][j];
		data_arr[i][temp] = 1;
	}
}

//void extractImages(float*** data_arr, double** arr, int rows, int n_input_units)
//{	
//	#pragma omp parallel for
//	for (int i = 0; i < rows; i++)
//	{
//		for (int j = 1; j < n_input_units; j++)		//erste node immer 1
//		{
//			data_arr[i][0][j] = (float)(arr[i][j] / 255);			//zwischen 0 ... 1 
//		}
//	}
//}

void extractImages(float*** data_arr, double** arr, int rows, int n_input_units)
{
#pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < n_input_units; j++)		//erste node immer 1
		{
			data_arr[i][0][j] = (float)(arr[i][j] / 255);			//zwischen 0 ... 1 
		}
	}
}

int load_mnist(struct net* net, int row, const char* path, float** label_data, float*** nodes)
{
	//double** data;
	//data = (double**)malloc(row * sizeof(double*));
	//for (int i = 0; i < row; ++i) {
	//	data[i] = (double*)malloc(net->n_nodes[0] * sizeof(double));
	//}
	read_csv(row, /*N_INPUT_UNITS,*/ path, net->mnist_data);
	extractLabel(label_data, net->mnist_data, row);
	extractImages(nodes, net->mnist_data, row, net->n_nodes[0]);
	
	//for (int i = 0; i < row; i++)
	//{
	//	free(data[i]);
	//}
	//free(data);

	return 0;
}

void testLoadData(struct net* net, struct stp* stp)
{

	//net->nodes[0][0][0] = 1;
	net->nodes[0][0][1] = 0.7;
	net->nodes[0][0][2] = 0.6;

	net->train_label_data[0][0] = 0.9;
	net->train_label_data[0][1] = 0.2;

	net->weights[0][0][0] = 0.3;
	net->weights[0][0][1] = 0.8;
	net->weights[0][0][2] = 0.5;
	net->weights[0][1][0] = -0.2;
	net->weights[0][1][1] = -0.6;
	net->weights[0][1][2] = 0.7;

	net->weights[1][0][0] = 0.2;
	net->weights[1][0][1] = 0.4;
	net->weights[1][0][2] = 0.3;
	net->weights[1][1][0] = 0.1;
	net->weights[1][1][1] = -0.4;
	net->weights[1][1][2] = 0.9;
	printf("\nTest Daten\n");
	feedForward(net, stp, stp->n_train, net->nodes);
	backProp(net, stp, 0);
}

void read_csv(int row, /*int col,*/ const char* filename, double** data) {
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

void reCallocNodesRows(struct stp* stp, struct net* net, int n_old, int n_new)
{
	//free
	float* current_sing_ptr;
	float** current_doub_ptr;
	for (int i = 0; i < n_old; i++)
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
	//calloc
	net->nodes = (float***)calloc(n_new, sizeof(float**));
	for (int i = 0; i < n_new; i++)
	{
		net->nodes[i] = (float**)calloc(stp->n_layer, sizeof(float*));
		for (int j = 0; j < stp->n_layer; j++)
		{
			net->nodes[i][j] = (float*)calloc(net->n_nodes[j], sizeof(float));
			net->nodes[i][j][0] = 1;			//der bias 
		}
	}
}

void testNeuralNet(struct net* net, struct stp* stp)
{
	int* predictions;
	predictions = (int*)calloc(stp->n_test, sizeof(int));
	int* labels;
	labels = (int*)calloc(stp->n_test, sizeof(int));
	float mse = 0;
	int prediction;
	feedForward(net, stp, stp->n_test, net->test_nodes);

	#pragma omp parallel for simd
	for (int test_row = 0; test_row < stp->n_test; test_row++)
	{
		prediction = 0;
		//printf("ROW: %d\n", test_row);
		for (int i = 0; i < stp->n_outputs; i++)
		{
			//printf("Zahl %d => pred:  %f| ist: %f\n", i, net->test_nodes[test_row][stp->n_layer-1][i], net->test_label_data[test_row][i]);
			if (net->test_nodes[test_row][stp->n_layer-1][prediction] <= net->test_nodes[test_row][stp->n_layer - 1][i])
			{
				prediction = i;
			}
			if (net->test_label_data[test_row][i] == 1)
			{
				labels[test_row] = i;
			}
			//printf("\nZahl %d => pred:  %f| ist: %f",i, net->nodes[test_row][stp->n_layer-1][i], net->test_label_data[test_row][i]);
		}
		predictions[test_row] = prediction;
		//printf("Pred = %d| Label = %d\n", prediction, labels[test_row]);
		
	}
	#pragma omp parallel for simd
	for (int j = 0; j < stp->n_test; j++)
	{
		if (predictions[j] == labels[j])
		{
			#pragma omp atomic
			mse++;
			//printf("\n%.2lf\n", mse);
		}
	}
	mse = (mse / stp->n_test) * 100;
	printf("%.2lf\n", mse);

	free(predictions);
	free(labels);
}

void trainNeuralNet(struct net* net, struct stp* stp)
{
	load_mnist(net, stp->n_train, train_data_path, net->train_label_data, net->nodes);
	load_mnist(net, stp->n_test, test_data_path, net->test_label_data, net->test_nodes);
	for (int i = 0; i < stp->epochs; i++)
	{
		printf("epoch: %d\n", i + 1);
		feedForward(net, stp, stp->n_train, net->nodes);
		for (int row = 0; row < stp->n_train; row++)
		{
			backProp(net, stp, row);
		}
		testNeuralNet(net, stp);
		//stp->learn_rate *= 0.99999;
		printf("\n");
	}
}
/*
		erstes Argument ist Epoche, zweites ist Learning rate
*/
int main(int argc, char* argv[])
{
	if (argc != 3) {
		exit(1);
	}
	struct net* net;
	struct stp* stp;
	net = (struct net*)malloc(sizeof(struct net));
	stp = (struct stp*)malloc(sizeof(struct stp));
	stp->n_test = 10000;
	stp->n_train = 60000;
	stp->n_inputs = 784;
	stp->n_outputs = 10;
	stp->n_hidden_nodes = 12; 
	stp->n_layer = 3;
	stp->epochs = strtol(argv[1], NULL, 10);
	stp->learn_rate = atof(argv[2]);
	initNet(net,stp);
	trainNeuralNet(net, stp);
	//deInitNet(net, stp);  funktioniert noch nicht

	printf("\n");

	free(net);
	free(stp);
	return 0;
}
