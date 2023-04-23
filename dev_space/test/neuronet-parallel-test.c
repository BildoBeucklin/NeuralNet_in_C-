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
	int batch_size;
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
	float** train_label_data;
	float** test_label_data;
	float** biases;
	/*
		weights[layer in dem wir uns befinden][knoten des darauffolgenden layer][die knoten in diesem layer]
		und immer von vorne nach hinten, also weights[0] beinhaltet die flanken die auf den input zeigen und in
		der letzten Schich sind die Flanken auf die der Output zeigt
	*/
	float*** weights; 
	float** learn_factors;
	float*** nodes; 				//alle Neruronen
	float*** test_nodes;
	int* n_nodes; 				//anzahl der Neuronen in jeder schicht von input bis output

} network;


void resetLearnFactors(struct net* net, struct stp* stp);
void testNeuralNet(struct net* net, struct stp* stp);
void initNet(struct net* net, struct stp* stp);
void deInitNet(struct net* net, struct stp* stp);
float dotProduct(float* a, float* b, int n);
float sigmoid(float x);
float derivedFunktion(float f);
void extractLabel(float** data_arr, double** arr, int n_input);
void extractImages(float*** data_arr, double** arr, int n_input, int n_input_units);
void read_csv(int row, const char* filename, double** data);
int load_mnist(struct net* net, int row, const char* path, float** label_data, float*** nodes);
void trainNeuralNet(struct net* net, struct stp* stp);
float reLU(float x);
void feedForward(struct net* net, struct stp* stp, int start, int ende, float*** nodes);
void deltaBackProp(struct net* net, struct stp* stp, int row);
void deltaShould(struct net* net, struct stp* stp, int row);
void backProp(struct net* net, struct stp* stp, int row);

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

	net->learn_factors = (float**)calloc(stp->n_layer-1, sizeof(float*));
	for (int i = 0; i < stp->n_layer-1; i++)
	{
		net->learn_factors[i] = (float*)calloc(net->n_nodes[i+1], sizeof(float));
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

	net->weights = (float***)calloc((stp->n_layer-1), sizeof(float**));
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

void deltaShould(struct net* net, struct stp* stp,int row)
{
	int out_layer = stp->n_layer - 1;
	for (int node = 0; node < net->n_nodes[out_layer]; node++)
	{
		net->learn_factors[out_layer - 1][node] = derivedFunktion(net->nodes[row][out_layer][node]) * -2 * (net->nodes[row][out_layer][node] - net->train_label_data[row][node]) ;
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
	#pragma omp parallel for simd
	for (int i = 0; i < stp->n_layer-1; i++)
	{
		for (int j = 0; j < net->n_nodes[i+1]; j++)
		{
			net->learn_factors[i][j] = 0;
		}
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

void testNeuralNet(struct net* net, struct stp* stp)
{
	int* predictions;
	predictions = (int*)calloc(stp->n_test, sizeof(int));
	int* labels;
	labels = (int*)calloc(stp->n_test, sizeof(int));
	float mse = 0;
	int prediction;
	feedForward(net, stp, 0, stp->n_test, net->test_nodes);

	#pragma omp parallel for simd
	for (int test_row = 0; test_row < stp->n_test; test_row++)
	{
		prediction = 0;
		for (int i = 0; i < stp->n_outputs; i++)
		{
			if (net->test_nodes[test_row][stp->n_layer-1][prediction] <= net->test_nodes[test_row][stp->n_layer - 1][i])
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
		int row = 0;
		while(row < stp->n_train)
		{
			feedForward(net, stp, row, row+stp->batch_size, net->nodes);
			for (int j = 0; j < stp->batch_size; j++)
		    {
			     backProp(net, stp, row+j);
	    	}
			row += stp->batch_size;
		}
		stp->learn_rate *= 0.999;
	}
	testNeuralNet(net, stp);
}

/*
		1. epochs, 2. learn rate, 3. batch size, 4. n test, 5. n train, 6. layer, 7. inputs, 8. hidden nodes, 9. outputs
		./neuronet-parallel-test 100 0.01 20 10000 60000 5 784 50 10
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
	initNet(net,stp);
	trainNeuralNet(net, stp);
	//deInitNet(net, stp);  funktioniert noch nicht
	free(net);
	free(stp);
	return 0;
}
