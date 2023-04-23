#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <stdbool.h>

//for windows wegen unistd.h
#ifdef WIN32
#include <io.h>
#define F_OK 0
#define access _access
#endif


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
	int learn_rate;

	int n_layer;
	int n_outputs;
	int n_inputs;

	int n_hidden_nodes;

	int act_func;
	int dataset;
	
} setup;


typedef struct net
{
	
	//float** train_image_data;
	float** train_label_data;
	//float** test_image_data;
	float** test_label_data;

	float** bias;
	/*float** outputs;*/
	/*float** test_outputs;*/
	float*** weights;   
	/*
		weights[layer in dem wir uns befinden][knoten des vorherigen layer][die knoten in diesem layer] 
		und immer von vorne nach hinten, also weights[0] beinhaltet die flanken die auf den input zeigen und in
		der letzten Schich sind die Flanken auf die der Output zeigt
	*/
	float*** nodes; 				//alle Neruronen
	int* n_nodes; 				//anzahl der Neuronen in jeder schicht von input bis output

} network;

void testNeuralNet(struct net* net, struct stp* stp, int n);
void initNet(struct net* net, struct stp* stp);
float dotProduct(float* a, float* b, int n);
float sigmoid(float x);
float derivedSigmoid(float x);
void extractLabel(float** data_arr, double** arr, int n_input);
void extractImages(float*** data_arr, double** arr, int n_input, int n_input_units);
void read_csv(int row, /*int col,*/ const char* filename, double** data);
int load_mnist(int row, const char* path, float** label_data, float*** image_data);
void trainNeuralNet(struct net* net, struct stp* stp);
//void deltaLearn(struct net* net, struct stp* stp, int output_layer);
void waitFor(unsigned int secs);
float reLU(float x);
void trainNeuralNet(struct net* net, struct stp* stp);
void feedForward(struct net* net, struct stp* stp, int x);
void backProp(struct net* net, struct stp* stp);

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
					net->weights[i][j][k] = ((float)rand() / (float)RAND_MAX)- ((float)rand() / (float)RAND_MAX);
				}
			}
		}
		else if (i>0 && i<stp->n_layer-1)
		{
			for (int j = 0; j < stp->n_hidden_nodes; j++)
			{
				for (int k = 0; k< stp->n_hidden_nodes; k++)
				{
					net->weights[i][j][k] = ((float)rand() / (float)RAND_MAX) - ((float)rand() / (float)RAND_MAX);
				}
			}
		}
		else
		{
			for (int j = 0; j < stp->n_outputs; j++)
			{
				for (int k = 0; k< stp->n_hidden_nodes; k++)
				{
					net->weights[i][j][k] = ((float)rand() / (float)RAND_MAX) - ((float)rand() / (float)RAND_MAX);
				}
			}			
		}		
	}
	
	net->n_nodes = (int*)calloc(stp->n_layer, sizeof(int));
	for(int i = 0; i < stp->n_layer; i++)
	{
		if(i==0)
		{
			net->n_nodes[i]=stp->n_inputs;
		}
		else if (i>0 && i<stp->n_layer-1)
		{
			net->n_nodes[i]=stp->n_hidden_nodes;
		}
		else
		{
			net->n_nodes[i]=stp->n_outputs;				
		}
	}
	
	net->nodes = (float***)malloc((stp->n_train)* sizeof(float**));
	for (int i = 0; i < stp->n_train; i++)
	{
		net->nodes[i] = (float**)malloc((stp->n_layer)* sizeof(float*));
		for (int j = 0; j < stp->n_layer; ++j)
		{
			net->nodes[i][j] = (float*)malloc((net->n_nodes[j])* sizeof(float));
		}
	}
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

float derivedSigmoid(float f) 		//mit sigmoid als input
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

void feedForward(struct net* net, struct stp *stp, int x)
{
	float temp;
	//#pragma omp parallel for
	for (int row = 0; row < x; row++)
	{
		for (int layer = 0; layer < stp->n_layer-1; layer++)
		{
			for (int node = 0; node < net->n_nodes[layer+1]; node++)
			{
				temp = dotProduct(net->nodes[row][layer], net->weights[layer][node], net->n_nodes[layer]);
				net->nodes[row][layer+1][node] = sigmoid(temp);  //+ bias[layer][node]
			}
		}
	}
}

/*
void deltaLearn(struct net* net, struct stp* stp, int output_layer)		//für die output layer
{
	for (int train_row = 0; train_row < stp->n_train; train_row++)
	{
		feedForward(net, stp, net->outputs, net->train_image_data, train_row);
		for (int i_output = 0; i_output < N_OUTPUTS; i_output++)
		{
			float learnFactor = stp->learn_rate * (net->train_label_data[train_row][i_output] - net->outputs[train_row][i_output]);
			for (int i_input = 0; i_input < N_INPUT_UNITS; i_input++)		//real delta learn
			{
				net->weights[output_layer][i_output][i_input] += net->train_image_data[train_row][i_input] * learnFactor;
			}
		}
	}
}
*/

void backProp(struct net* net, struct stp* stp)
{
	feedForward(net, stp, stp->n_train);
	float learn_factor;
	waitFor(1);
	if (stp->n_layer == 2) 			// dann deltaLearn
	{
		for (int train_row = 0; train_row < stp->n_train; train_row++)
		{
			for (int i = 0; i < stp->n_outputs; i++)
			{
				learn_factor = stp->learn_rate * (net->train_label_data[train_row][i] - net->nodes[train_row][1][i]);
				for (int j = 0; j < stp->n_inputs; j++)
				{
					net->weights[stp->n_layer - 1][i][j] += net->nodes[train_row][0][j] * learn_factor;
				}
			}
		}
	}
	else
	{
		for(int train_row = 0; train_row < stp->n_train; train_row++)
		{

			for(int layer = stp->n_layer-1; layer >= 0; layer --)
			{
				printf("\nwar da");
				if(layer == stp->n_layer-1)
				{
					for(int i = 0; i < net->n_nodes[layer]; i++)
					{																																/* ######## der output #########*/	
						learn_factor = stp->learn_rate * derivedSigmoid(net->nodes[train_row][layer-1][i]) * (net->train_label_data[train_row][i] - net->nodes[layer][train_row][i]);
						for(int j = 0; j < net->n_nodes[layer-1]; j++)
						{
							net->weights[layer][i][j] += net->nodes[train_row][layer][j] * learn_factor;
						}
					}
				}
				else
				{
					for (int j = 0; j < net->n_nodes[layer]; j++)
					{

					}
				}
			}
		}
		
	}
}

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

void extractImages(float*** data_arr, double** arr, int n_input, int n_input_units)
{	
	#pragma omp parallel for
	for (int i = 0; i < n_input; i++)
	{
		for (int j = OFFSET_IMAGE; j < n_input_units; j++)
		{
			data_arr[i][0][j - 1] = (float)(arr[i][j] / 255);			//zwischen 0 ... 1 
		}
	}
}

int load_mnist(int row, const char* path, float** label_data, float*** image_data)
{
	double** data;
	data = (double**)malloc(N_TRAIN_ROW * sizeof(double*));
	for (int i = 0; i < N_TRAIN_ROW; ++i) {
		data[i] = (double*)malloc((N_INPUT_UNITS + 1) * sizeof(double));
	}

	read_csv(row, /*N_INPUT_UNITS,*/ path, data);
	extractLabel(label_data, data, row);
	extractImages(image_data, data, row, N_INPUT_UNITS);
	
	return 0;
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

void testNeuralNet(struct net* net, struct stp* stp, int n)
{
	int* predictions = (int*)calloc(n, sizeof(int));
	int* labels = (int*)calloc(n, sizeof(int));
	float mse = 0;
	int prediction;

	feedForward(net, stp, n);

	//#pragma omp parallel for
	for (int test_row = 0; test_row < n; test_row++)
	{
		prediction = 0;
		//printf("\nROW: %d", test_row);
		for (int i = 0; i < stp->n_outputs; i++)
		{
			if (net->nodes[test_row][stp->n_layer-1][prediction] <= net->nodes[test_row][stp->n_layer - 1][i])
			{
				prediction = i;
			}
			if (net->test_label_data[test_row][i] == 1)
			{
				labels[test_row] = i;
			}
			//printf("\nZahl %d => pred:  %f| ist: %f",i, net->nodes[test_row][stp->n_layer - 1][i], net->test_label_data[test_row][i]);
		}
		predictions[test_row] = prediction;
		
	}
	//#pragma omp parallel for
	for (int j = 0; j < n; j++)
	{
		if (predictions[j] == labels[j])
		{
			//#pragma omp atomic
			mse++;
			//printf("\n%.2lf\n", mse);
		}
	}
	mse = (mse / stp->n_test) * 100;
	printf("\n%.2lf\n", mse);
}

void trainNeuralNet(struct net* net, struct stp* stp)
{
	for (int i = 0; i < stp->epochs; i++)
	{
		/*deltaLearn(net, stp);*/
		backProp(net, stp);
		stp->learn_rate *= 0.9;
		testNeuralNet(net, stp, stp->n_test);
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
	stp->n_test = 10;
	stp->n_train = 10;
	stp->n_inputs = N_INPUT_UNITS;
	stp->n_outputs = N_OUTPUTS;
	stp->n_hidden_nodes = 1000;
	stp->n_layer = 2;
	stp->epochs = strtol(argv[1], NULL, 10);
	stp->learn_rate = atof(argv[2]);
	initNet(net,stp);
	load_mnist(stp->n_train, train_data_path, net->train_label_data, net->nodes);
	load_mnist(stp->n_test, test_data_path, net->test_label_data, net->nodes);
	trainNeuralNet(net, stp);


	return 0;
}
