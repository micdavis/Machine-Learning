#include <cmath>
using namespace std;

double* backPropLayer(double** weights, double* biases, double* deltaCost, double* prevActivation, int weightsLen, int weightsLenTwo);
double sigmoid(double x);
double sigmoidDerivative(double x);
double matrixMultiplySum(double* inputs, double** weights, int indexInLayer, int weightsLen, int weigthLenTwo);


int main () {

	//defining layers of specific size
	double* inputLayer = new double[784];
	double* hiddenLayer = new double[15];
	double* outputLayer = new double[10];

	//defining biases of specific size
	double* hiddenLayerBias = new double[15];
	double* outputLayerBias = new double[10];

	//defining weights of specific size
	double** weightsOne = new double* [784];
	for(int i = 0; i < 784; i++)
	{
		weightsOne[i] = new double[15];
	}

	double** weightsTwo = new double* [15];
	for(int i = 0; i < 15; i++)
	{
		weightsTwo[i] = new double[10];
	}

	//number of learning cycles
	for(int i = 0; i < 100; i++)
	{
		double** weightsOneDelta = new double* [784];
		for(int j = 0; j < 784; j++)
		{
			weightsOneDelta[j] = new double[15];
		}

		double* hiddenBiasDelta = new double [15];

		double** weightsTwoDelta = new double* [15]; 
		for(int j = 0; j < 15; j++)
		{
			weightsTwoDelta[j] = new double[10];
		}

		double* outputBiasDelta = new double[10];
		//number of test cases ran through
		for(int j = 0; j < 100; j++)
		{
			//set input layers to an input and expected output
			double* expectedOutput = new double[10];

			//forward propagation
			for(int k = 0; k < 15; k++)
			{
				hiddenLayer[k] = sigmoid(matrixMultiplySum(inputLayer, weightsOne, k, 784, 15) + hiddenLayerBias[k]);
			}
			for(int k = 0; k < 10; k++)
			{
				outputLayer[k] = sigmoid(matrixMultiplySum(hiddenLayer, weightsTwo, k, 15, 10) + outputLayerBias[k]);
			}

			//on to the back prop
			double* deltaCost = backPropLayer(weightsTwo, outputLayerBias, expectedOutput, hiddenLayer, 784, 15);
			backPropLayer(weightsOne, hiddenLayerBias, deltaCost, inputLayer, 15, 10);
			delete[] deltaCost;
		}
	}
}


double matrixMultiplySum(double* inputs, double** weights, int indexInLayer, int weightsLen, int weightsLenTwo)
{
	double sum = 0;
	for(int i = 0; i < weightsLen; i++)
	{
		sum += inputs[i] * weights[i][indexInLayer];
	}
	return sum;
}

//Used for forward prop
double sigmoid(double x)
{
	return 1/(1 + pow(M_E, -x));
}

double* backPropLayer(double** weights, double* biases, double* deltaCost, double* prevActivation, int weightsLen, int weightsLenTwo)
{
	double* deltaNextActivation = new double[weightsLen];
	double* deltaBiases = new double[sizeof biases/sizeof(double)];
	double** deltaWeights = new double* [weightsLen];
	for(int i = 0; i < weightsLen; i++)
	{
		deltaWeights[i] = new double [weightsLenTwo];
	}

	//setting the deltas
	for(int i = 0; i < (sizeof biases/sizeof(double)); i++)
	{
		for(int j = 0; j < weightsLen; j++)
		{
			deltaBiases[i] += sigmoidDerivative(prevActivation[j] * weights[j][i] + biases[i]) * deltaCost[i];
			deltaWeights[j][i] = prevActivation[j] * sigmoidDerivative(prevActivation[j] * weights[j][i] + biases[i]) * deltaCost[i];
			deltaNextActivation[j] += weights[j][i] * sigmoidDerivative(prevActivation[j] * weights[j][i] + biases[i]) * deltaCost[i];
		}
	}

	//applying the deltas to the neural net
	for(int i = 0; i < (sizeof biases/sizeof(double)); i++)
	{
		biases[i] += biases[i];
		for(int j = 0; j < weightsLen; j++)
		{
			weights[j][i] += deltaWeights[j][i];
		}
	}

	//return for next layer of backprop
	return deltaNextActivation;
}

//Used for back prop
double sigmoidDerivative(double x)
{
	return pow(M_E, -x)/pow(1 + pow(M_E, -x), 2);
}