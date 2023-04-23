#include "inc/Python/Python.h" 
#include "inc/neuronet.h"

double pyMnsitNet(int epochs, float learn_rate, int batch_size, int n_layer, int n_hidden)
{
	//struct net* net;
	//struct stp* stp;
	//net = (struct net*)malloc(sizeof(struct net));
	//stp = (struct stp*)malloc(sizeof(struct stp));
	//stp->epochs = epochs;
	//stp->learn_rate = learn_rate;
	//stp->batch_size = batch_size;
	//stp->n_test = 10000;
	//stp->n_train = 60000;
	//stp->n_layer = n_layer;
	//stp->n_inputs = 784;
	//stp->n_hidden_nodes = n_hidden;
	//stp->n_outputs = 10;
	//initNet(net, stp);
	//trainMnist(net, stp);
	//float mse = 0.0;
	//mse = testNet(net, stp);
	//printf("%f", mse);
	//free(net);
	//free(stp);


	return mse;
}

#define PY_SSIZE_T_CLEAN

/*wrapping funktion fuer python module*/
static PyObject* py_MnistNet(PyObject* self, PyObject* args)
{
	int* epochs, * batch_size, * n_layer, * n_hidden = NULL;
	float learn_rate = 0.001;

	/* Parse arguments */
	if (!PyArg_ParseTuple(args, "iiii", &epochs, &batch_size, &n_layer, &n_hidden)) {
		return NULL;
	}
	/*die eigentliche Funktion*/

	double mse = pyMnsitNet(epochs, learn_rate, batch_size, n_layer, n_hidden);

	return PyFloat_FromDouble(mse);
}


/*sagt dem python interpreter was das hier ist gut wenn mehr als eine funktion hier drin steht*/
static PyMethodDef NetMethods[] = {
	{"pyMnistNet",  py_MnistNet, METH_VARARGS, "Neuronal Net for Python"},
	{NULL, NULL, 0, NULL}
};

/*entaelt die information ueber die methode*/
static struct PyModuleDef pyMnistNetmodule = {
	PyModuleDef_HEAD_INIT,
	"pyMNISTNet",
	"Neuronal Net for Python",
	-1,
	NetMethods
};

/*wenn python das module importiert*/
PyMODINIT_FUNC PyInit_pyMnistNet(void) {
	return PyModule_Create(&pyMnistNetmodule);
}
