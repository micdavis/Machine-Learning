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

			//this is the output layer biases and weights
			for(int k = 0; k < 10; k++)
			{
				for(int l = 0; l < 15; l++)
				{
					weightsTwoDelta[l][k] += tweakWeight(hiddenLayer[l], matrixMultiplySum(hiddenLayer, weightsTwo, k) + outputLayerBias[k], expectedOutput[k]);
				}
				outputBiasDelta[k] += tweakBias(matrixMultiplySum(hiddenLayer, weightsTwo, k) + outputLayerBias[k], expectedOutput[k]);
			}

			//getting the expective activation for the hidden layer
			double expectedHidden = new double[15];
			for(int k = 0; k < 10; k++)
			{
				for(int l = 0; l < 15; l++)
				{
					expectedHidden[l] += expectedActiviation(weightsTwo[l][k], matrixMultiplySum(hiddenLayer, weightsTwo, k) + outputLayerBias[k], expectedOutput[k]);
				}
			}

			for(int k = 0; k < 15; k++)
			{
				expectedHidden[k] = expectedHidden[k] / 10.0;
			}

			//repeat above for the hidden later biases and weights
			for(int k = 0; k < 15; k++)
			{
				for(int l = 0; l < 784; l++)
				{
					weightsOneDelta[l][k] += tweakWeight(inputLayer[l], matrixMultiplySum(inputLayer, weightsOne, k) + hiddenLayerBias[k], expectedHidden[k]);
				}
				hiddenLayerBias[k] += tweakBias(matrixMultiplySum(inputLayer, weightsOne, k) + hiddenLayerBias[k], expectedHidden[k]);
			}
		}

		//average all the delta biases and weights and apply them to the NN
		for(int j = 0; j < 784; j++)
		{
			for(int k = 0; k < 15; k++)
			{
				weightsOneDelta[j][k] = weightsOneDelta[j][k] / 100.0;
				weightsOne[j][k] += weightsOneDelta[j][k];
			}
		}

		for(int j = 0; j < 15; j++)
		{
			hiddenBiasDelta[j] = hiddenBiasDelta[j] / 100.0;
			hiddenLayerBias[j] = hiddenBiasDelta[j];
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


//Used for back prop
double sigmoidDerivative(double x)
{
	return pow(M_E, -x)/pow(1 + pow(M_E, -x), 2)
}

double calcCost(double actual[], double expected[])
{
	double cost = 0;
	for(int i = 0; i < actual.len; i++)
	{
		cost += pow(actual - expected, 2);
	}
	return cost;
}

double tweakWeight(double prevActivation, double currentNeuronVal, double expected)
{
	return prevActivation * tweakBias(currentNeuronVal, expected);
}

double tweakBias(double currentNeuronVal, double expected)
{
	return sigmoidDerivative(currentNeuronVal) * 2 * (sigmoid(currentNeuronVal) - expected);
}

double expectedActiviation(double weight, double currentNeuronVal, double expected)
{
	return weight * tweakBias(currentNeuronVal, expected);
}