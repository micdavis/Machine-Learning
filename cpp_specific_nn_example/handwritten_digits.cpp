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

	//number of test cases ran through
	for(int i = 0; i < 10000; i++)
	{
		//set input layers to an input and expected output
		double expected = new double[10];
		double actual = new double[10]; 

		//forward propagation
		for(int i = 0; i < 15; i++)
		{
			hiddenLayer[i] = sigmoid(matrixMultiplySum(inputLayer, weightsOne) + hiddenLayerBias[i]);
		}
		for(int i = 0; i < 10; i++)
		{
			outputLayer[i] = sigmoid(matrixMultiplySum(hiddenLayer, weightsTwo) + outputLayerBias[i]);
		}

		double cost = cost(actual expected);

		
	}
}


double matrixMultiplySum(double inputs[], double weights[])
{
	double sum = 0;
	for(int i = 0; i < inputs.len; i++)
	{
		sum += imputs[i] * weights[i];
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