#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <time.h>

//f�r windows wegen unistd.h
#ifdef WIN32
#include <io.h>
#define F_OK 0
#define access _access
#endif

#define N_TRAIN_ROW 60000				//gerade nur 10000 wieder auf 60000 setzen
#define N_TEST_ROW 10000
#define N_INPUT_UNITS 784				//28*28 Anzahl der Pixel
#define N_NODES	794						//alle Knoten au�er input layer hidden nodes + 10 output node
#define OFFSET_IMAGE 1
#define N_OUTPUTS 10 

static const char train_data_path[] = "./data/mnist_train.csv";
static const char test_data_path[] = "./data/mnist_test.csv";

typedef struct net
{
	float** train_image_data;
	float** train_label_data;
	float** test_image_data;
	float** test_label_data;

	float** outputs;
	float** test_outputs;
	float** weights;

	float mse;
	float learn_rate;
	int epochs;								// wie oft man das komplette dataset einlie�t

} network;

void testNeuralNet(struct net* net);
void initNet(struct net* net);
float dotProduct(float** a, float** b, int n, int x, int j);
float sigmoid(float x);
void mseLoss(struct net* net, float** train_labels, float** prediction, int j);
float derivedSigmoid(float x);
void extractLabel(float** data_arr, double** arr, int n_input);
void extractImages(float** data_arr, double** arr, int n_input, int n_input_units);
void read_csv(int row, int col, const char* filename, double** data);
int load_mnist(int row, const char* path, float** label_data, float** image_data);
void trainNeuralNet(struct net* net);
void deltaLearn(struct net* net);
void waitFor(unsigned int secs);
float reLU(float x);
void trainNeuralNet(struct net* net);
char* concat(const char* s1, const char* s2);
void feedForward(struct net* net, float** outputs, float** image_data, int n_row, int n_col);

void initNet(struct net* net)
{
	net->outputs = (float**)calloc(N_TRAIN_ROW, sizeof(float*));
	for (int i = 0; i < N_TRAIN_ROW; ++i)
	{
		net->outputs[i] = (float*)calloc(N_OUTPUTS, sizeof(float));
	}

	net->test_outputs = (float**)calloc(N_TRAIN_ROW, sizeof(float*));
	for (int i = 0; i < N_TRAIN_ROW; ++i)
	{
		net->test_outputs[i] = (float*)calloc(N_OUTPUTS, sizeof(float));
	}

	net->train_label_data = (float**)calloc(N_TRAIN_ROW, sizeof(float*));
	for (int i = 0; i < N_TRAIN_ROW; ++i)
	{
		net->train_label_data[i] = (float*)calloc(N_OUTPUTS, sizeof(float));
	}

	net->train_image_data = (float**)calloc(N_TRAIN_ROW, sizeof(float*));
	for (int i = 0; i < N_TRAIN_ROW; ++i)
	{
		net->train_image_data[i] = (float*)calloc(N_INPUT_UNITS, sizeof(float));
	}

	net->test_label_data = (float**)calloc(N_TEST_ROW, sizeof(float*));
	for (int i = 0; i < N_TEST_ROW; ++i)
	{
		net->test_label_data[i] = (float*)calloc(N_OUTPUTS, sizeof(float));
	}

	net->test_image_data = (float**)calloc(N_TEST_ROW, sizeof(float*));
	for (int i = 0; i < N_TEST_ROW; ++i)
	{
		net->test_image_data[i] = (float*)calloc(N_INPUT_UNITS, sizeof(float));
	}

	net->weights = (float**)calloc(N_OUTPUTS, sizeof(float*));
	for (int i = 0; i < N_OUTPUTS; ++i)
	{
		net->weights[i] = (float*)calloc(N_INPUT_UNITS, sizeof(float));
	}
	for (int i = 0; i < N_OUTPUTS; i++)
	{
		for (int j = 0; j < N_INPUT_UNITS; j++)
		{
			net->weights[i][j] = ((float)rand() / (float)RAND_MAX);
		}
	}
}

void waitFor(unsigned int secs) {
	unsigned int retTime = time(0) + secs;   // Get finishing time.
	while (time(0) < retTime);               // Loop until it arrives.
}

char* concat(const char* s1, const char* s2)
{
	char* result = malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

float dotProduct(float** a, float** b, int n, int b_row, int a_row)
{
	float sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += a[a_row][i] * b[b_row][i];
	}
	return sum;
}

void mseLoss(struct net* net, float** train_labels, float** prediction, int j)
{
	net->mse = 0;
	for (int i = 0; i < N_OUTPUTS; i++)
	{
		net->mse = +pow((train_labels[j][i] - prediction[j][i]), 2);
	}
}

float sigmoid(float x)
{
	float f;
	f = 1 / (1 + exp(-x));
	return f;
}

float derivedSigmoid(float x)
{
	float f = sigmoid(x);
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

void feedForward(struct net* net, float** outputs, float** image_data, int row, int n_col)
{
	for (int col = 0; col < n_col; col++)
	{
		outputs[row][col] = dotProduct(net->weights, image_data, N_INPUT_UNITS, row, col);
		outputs[row][col] = sigmoid(outputs[row][col]);
	}
}

void deltaLearn(struct net* net)
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
				printf("\nweights[%d][%d] = %f",i_output, i_input, net->weights[i_output][i_input]);
			}

		}
	}
}

void extractLabel(float** data_arr, double** arr, int n_input)	
{
	int temp;
	int j = 0;
	for (int i = 0; i < n_input; i++)
	{
		temp = arr[i][j];
		data_arr[i][temp] = 1;
	}
}

void extractImages(float** data_arr, double** arr, int n_input, int n_input_units)
{
	for (int i = 0; i < n_input; i++)
	{
		for (int j = OFFSET_IMAGE; j < n_input_units; j++)
		{
			data_arr[i][j - 1] = (float)(arr[i][j] / 255);			//zwischen 0 ... 1 
		}
	}
}

int load_mnist(int row, const char* path, float** label_data, float** image_data)
{
	double** data;
	data = (double**)malloc(N_TRAIN_ROW * sizeof(double*));
	for (int i = 0; i < N_TRAIN_ROW; ++i) {
		data[i] = (double*)malloc((N_INPUT_UNITS+1) * sizeof(double));
	}
	read_csv(row, N_INPUT_UNITS, path, data);
	extractLabel(label_data, data, row);
	extractImages(image_data, data, row, N_INPUT_UNITS);
	return 0;
}

void read_csv(int row, int col, const char* filename, double** data) {
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

void testNeuralNet(struct net* net)		
{
	int prediction = 0;
	int label = 0;
	net->mse = 0;

	for (int test_row = 0; test_row < N_TEST_ROW; test_row++)
	{
		for (int i = 0; i < N_OUTPUTS; i++)
		{
			net->test_outputs[test_row][i] = dotProduct(net->weights, net->test_image_data, N_INPUT_UNITS, test_row, i);
			net->test_outputs[test_row][i] = sigmoid(net->test_outputs[test_row][i]);
			if (net->test_outputs[test_row][prediction] <= net->test_outputs[test_row][i])
			{
				prediction = i;
			}
			if (net->test_label_data[test_row][i] == 1)
			{
				label = i;
			}
		}
		if (prediction == label)
		{
			net->mse++;
		}
	}
	net->mse = (net->mse / N_TEST_ROW)*100;
	printf("%.2lf", net->mse);
}

void trainNeuralNet(struct net* net)
{
	for (int i = 0; i < net->epochs; i++)
	{
		deltaLearn(net);
		net->learn_rate *= 0.9;
	}
}
/*
		erstes Argument ist Epoche, zweites ist Learning rate
*/
int main(int argc, char* argv[])
{
	
	if(argc != 3){
		exit(1);
	}
	struct net* net;
	net = (struct net*)malloc(sizeof(struct net));
	net->epochs = strtol(argv[1], NULL, 10);
	net->learn_rate = atof(argv[2]);
	initNet(net);
	load_mnist(N_TRAIN_ROW, train_data_path, net->train_label_data, net->train_image_data);
	load_mnist(N_TEST_ROW, test_data_path, net->test_label_data, net->test_image_data);
	trainNeuralNet(net);
	testNeuralNet(net);
}
