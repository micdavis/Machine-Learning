#include <cmath>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <cctype>
#include <unistd.h>
using namespace std;

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0') 



double* backPropLayer(double** weights, double* biases, double* deltaCost, double* prevActivation, int weightsLen, int weigthLenTwo);
double sigmoid(double x);
double sigmoidDerivative(double x);
double matrixMultiplySum(double* inputs, double** weights, int indexInLayer, int weightsLen);


double calcError(double* expected, double* actual)
{
	double sum = 0.0;
	for(int i = 0; i < 10; i++)
	{
		sum += (actual[i] - expected[i]) * (actual[i] - expected[i]);
	}
	return sum / 10.0;
}

int main () {
	//defining layers of specific sizeof
	double* inputLayer = new double[784];
	double* hiddenLayer = new double[15];
	double* outputLayer = new double[10];

	//defining biases of specific size
	double* hiddenLayerBias = new double[15];
	double* outputLayerBias = new double[10];

	double max = 0.0;
	int maxindex = 0;
	int expectedindex = 0;
	//defining weights of specific size
	double** weightsOne = new double* [784];
	for(int i = 0; i < 784; i++)
	{
		weightsOne[i] = new double[15];
		for(int j = 0; j < 15; j++)
		{
			weightsOne[i][j] = (double)(rand() % 20 - 10);
		}
	}

	double** weightsTwo = new double* [15];
	for(int i = 0; i < 15; i++)
	{
		weightsTwo[i] = new double[10];
		for(int j = 0; j < 10; j++)
		{
			weightsTwo[i][j] = (double)(rand() % 20 - 10);
		}
	}
	//number of learning cycles
	for(int i = 0; i < 10; i++)
	{

		ifstream dataFile;
		ifstream labelFile;

		dataFile.open("/home/michael/Documents/gitRepos/Machine-Learning/cpp_specific_nn_example/trainingData/train-images-idx3-ubyte");
		labelFile.open("/home/michael/Documents/gitRepos/Machine-Learning/cpp_specific_nn_example/trainingData/train-labels-idx1-ubyte");

		for(int j = 0; j < 16; j++)
		{
			char in;
			if(j < 8)
			{
				labelFile >> in;
			}
			dataFile >> in;
			//printf("Leading text "BYTE_TO_BINARY_PATTERN, BYTE_TO_BINARY(in));
			//std:cout << in << std::endl;
		}		

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

		double error = 0.0;
		//number of test cases ran through
		for(int j = 0; j < 2000; j++)
		{
			//set input layers to an input and expected output
			double* expectedOutput = new double[10];

			char in;
			labelFile.get(in);
			//printf("Leading text "BYTE_TO_BINARY_PATTERN, BYTE_TO_BINARY(in));
			expectedindex = (int) in;
			expectedOutput[(int)in] = 1.0;

			//std::cout << (int)in << std::endl;

			for(int k = 0; k < 784; k++)
			{
				char in;
				dataFile.get(in);	
				//std::cout << (std::byte)in << std::endl;			
				inputLayer[k] = in;			
				//std::cout << k << ": " << hex2int(in) << std::endl;
			}

			double* deltaCost1 = new double[10];

			//forward propagation
			for(int k = 0; k < 15; k++)
			{
				hiddenLayer[k] = sigmoid(matrixMultiplySum(inputLayer, weightsOne, k, 784) + hiddenLayerBias[k]);
			}
			for(int k = 0; k < 10; k++)
			{
				outputLayer[k] = sigmoid(matrixMultiplySum(hiddenLayer, weightsTwo, k, 15) + outputLayerBias[k]);
				if(i == 9)
				{
					std::cout << outputLayer[k] << " : " << expectedOutput[k] << std::endl;
					if(outputLayer[k] > max) 
					{
						max = outputLayer[k];
						maxindex = k;
					}
				}
				deltaCost1[k] = 2.0	 * (outputLayer[k] - expectedOutput[k]);
			}
			max = 0;

			if(i == 9)
			{
				//std::cout << "A: " << maxindex << "E: " << expectedindex << std::endl;				
			}

			error += calcError(expectedOutput, outputLayer);
			//on to the back prop
			//std::cout << j << ": " << calcError(expectedOutput, outputLayer)  << std::endl;
			double* deltaCost = backPropLayer(weightsTwo, outputLayerBias, deltaCost1, hiddenLayer, 15, 10);
			backPropLayer(weightsOne, hiddenLayerBias, deltaCost, inputLayer, 784, 15);
			delete[] deltaCost;
			delete[] deltaCost1;
			delete[] expectedOutput;
		}
		std::cout << "Round " << i << " error: " << error/2000.0 << std::endl;
		delete[] weightsOneDelta;
		delete[] weightsTwoDelta;
		delete[] hiddenBiasDelta;
		delete[] outputBiasDelta;
	}
}


double matrixMultiplySum(double* inputs, double** weights, int indexInLayer, int weightsLen)
{
	double sum = 0.0;
	for(int i = 0; i < weightsLen; i++)
	{
		sum += inputs[i] * weights[i][indexInLayer];
		//std::cout << "SUM: " << inputs[i] << " + " << weights[i][indexInLayer] << "  " << indexInLayer << endl;
	}
	//std::cout << "SUM: " << sum << endl;
	return sum;
}

//Used for forward prop
double sigmoid(double x)
{
	return 1.0/(1.0 + pow(M_E, -x));
}

double* backPropLayer(double** weights, double* biases, double* deltaCost, double* prevActivation, int weightsLen, int weightsLenTwo)
{
	double* deltaNextActivation = new double[weightsLen];
	double* deltaBiases = new double[weightsLenTwo];
	double** deltaWeights = new double* [weightsLen];
	for(int i = 0; i < weightsLen; i++)
	{
		deltaWeights[i] = new double [weightsLenTwo];
	}

	//setting the deltas
	for(int i = 0; i < weightsLenTwo; i++)
	{
		for(int j = 0; j < weightsLen; j++)
		{
			deltaBiases[i] += sigmoidDerivative(prevActivation[j] * weights[j][i] + biases[i]) * deltaCost[i];
			deltaWeights[j][i] = prevActivation[j] * sigmoidDerivative(prevActivation[j] * weights[j][i] + biases[i]) * deltaCost[i];
			deltaNextActivation[j] += weights[j][i] * sigmoidDerivative(prevActivation[j] * weights[j][i] + biases[i]) * deltaCost[i];
		}
	}

	//applying the deltas to the neural net
	for(int i = 0; i < weightsLenTwo; i++)
	{
		if(deltaBiases[i] == -0) deltaBiases[i] = 0;
		biases[i] += deltaBiases[i];
		//std::cout << "deltaBiases: " << i << " : " << deltaBiases[i] << std::endl;
		for(int j = 0; j < weightsLen; j++)
		{
			if(deltaWeights[j][i] == -0) deltaWeights[j][i] = 0;
			weights[j][i] += deltaWeights[j][i];
			//std::cout << "deltaWeights: " << j << "  " << i << " : " << deltaWeights[j][i] << std::endl;
		}
	}

	delete[] deltaBiases;
	delete[] deltaWeights;

	//return for next layer of backprop
	return deltaNextActivation;
}

//Used for back prop
double sigmoidDerivative(double x)
{
	return sigmoid(x) * (1.0 - sigmoid(x));
}