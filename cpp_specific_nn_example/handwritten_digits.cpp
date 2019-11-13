#include <cmath>
using namespace std;

int main () {

	//defining layers of specific size
	double inputLayer[784] = new double[784];
	double hiddenLayer[15] = new double[15];
	double outputLayer[10] = new double[10];

	//defining biases of specific size
	double hiddenLayerBias[15] = new double[15];
	double outputLayerBias[10] = new double[10];

	//defining weights of specific size
	double weightsOne[784][15] = new double [784][15];
	double weightsTwo[15][10] = new double [15][10];

	//number of learning cycles
	for(int i = 0; i < 100; i++)
	{
		double weightsOneDelta[784][15] = new double [784][15];
		double hiddenBiasDelta[15] = new double [15];
		double weightsTwoDelta[15][10] = new double [15][10]; 
		double outputBiasDelta[10] = new double[10];
		//number of test cases ran through
		for(int j = 0; j < 100; j++)
		{
			//set input layers to an input and expected output
			double expectedOutput = new double[10];

			//forward propagation
			for(int k = 0; k < 15; k++)
			{
				hiddenLayer[k] = sigmoid(matrixMultiplySum(inputLayer, weightsOne, k) + hiddenLayerBias[k]);
			}
			for(int k = 0; k < 10; k++)
			{
				outputLayer[k] = sigmoid(matrixMultiplySum(hiddenLayer, weightsTwo, k) + outputLayerBias[k]);
			}

			//on to the back prop
			double* deltaCost = backPropLayer(weightsTwo, outputLayerBias, expectedOutput, hiddenLayer);
			backPropLayer = backPropLayer(weightsOne, hiddenLayerBias, deltaCost, inputLayer);
			delete[] deltaCost;
		}
	}
}


double matrixMultiplySum(double inputs[], double weights[][], int indexInLayer)
{
	double sum = 0;
	for(int i = 0; i < weights.len; i++)
	{
		sum += inputs[i] * weights[i][indexInLayer];
	}
	return sum;
}

//Used for forward prop
double sigmoid(double x)
{
	return 1/(1 + pow(M_E, -x))
}

double* backPropLayer(double weights[][], double biases[], double deltaCost[], double prevActivation[])
{
	double deltaNextActivation[weights.len] = new double[weights.len];
	double deltaBiases[biases.len] = new double[biases.len];
	double deltaWeights[weights.len][weights[0].len] = new double[weights.len][weights[0].len];

	//setting the deltas
	for(int i = 0; i < biases.len; i++)
	{
		for(int j = 0; j < weights.len; j++)
		{
			deltaBiases[i] += sigmoidDerivative(prevActivation[j] * weights[j][i] + biases[i]) * deltaCost[i];
			deltaWeight[j][i] = prevActivation[j] * sigmoidDerivative(prevActivation[j] * weights[j][i] + biases[i]) * deltaCost[i];
			deltaNextActivation[j] += weights[j][i] * sigmoidDerivative(prevActivation[j] * weights[j][i] + biases[i]) * deltaCost[i];
		}
	}

	//applying the deltas to the neural net
	for(int i = 0; i < biases.len; i++)
	{
		biases[i] += biases[i];
		for(int j = 0; j < weights.len; j++)
		{
			weights[j][i] += deltaWeight[j][i];
		}
	}

	//return for next layer of backprop
	return deltaNextActivation;
}

//Used for back prop
double sigmoidDerivative(double x)
{
	return pow(M_E, -x)/pow(1 + pow(M_E, -x), 2)
}