#include <cmath>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <cctype>
#include <unistd.h>
#include <time.h>
#include <thread>
using namespace std;

double* backPropLayer(double** weights, double* biases, double* deltaCost, double* prevActivation, int weightsLen, int weigthLenTwo, double* deltaNextActivation, double* nextActivation);
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
	srand(time(NULL));
	//defining layers of specific sizeof
	double* inputLayer = new double[784];
	double* hiddenLayer = new double[125];
	double* outputLayer = new double[10];

	//defining biases of specific size
	double* hiddenLayerBias = new double[125];
	double* outputLayerBias = new double[10];

	double max = 10.0;
	int maxIndex = 0;
	int expectedindex = 0;
	//defining weights of specific size
	double** weightsOne = new double* [784];
	for(int i = 0; i < 784; i++)
	{
		weightsOne[i] = new double[125];
		for(int j = 0; j < 125; j++)
		{
			weightsOne[i][j] = (double)(rand() % 20 - 10);
		}
	}

	double** weightsTwo = new double* [125];
	for(int i = 0; i < 125; i++)
	{
		weightsTwo[i] = new double[10];
		for(int j = 0; j < 10; j++)
		{
			weightsTwo[i][j] = (double)(rand() % 20 - 10);
		}
	}
	//number of learning cycles
	for(int i = 0; i < 100; i++)
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
		}		

		double** weightsOneDelta = new double* [784];
		for(int j = 0; j < 784; j++)
		{
			weightsOneDelta[j] = new double[125];
		}

		double* hiddenBiasDelta = new double [125];
		for(int q = 0; q < 125; q++)
		{
			hiddenBiasDelta[q] = 0;
		}

		double** weightsTwoDelta = new double* [125]; 
		for(int j = 0; j < 125; j++)
		{
			weightsTwoDelta[j] = new double[10];
		}

		double* outputBiasDelta = new double[10];
		for(int q = 0; q < 10; q++)
		{
			outputBiasDelta[q] = 0;
		}
		double error = 0.0;

		double score = 0.0;

		//number of test cases ran through
		for(int j = 0; j < 1001; j++)
		{
			//set input layers to an input and expected output
			double* expectedOutput = new double[10];
			for(int q = 0; q < 10; q++)
			{
				expectedOutput[q] = 0;
			}

			char in;
			labelFile.get(in);
			expectedindex = (int) in;
			expectedOutput[(int)in] = 1.0;

			for(int k = 0; k < 784; k++)
			{
				char in;
				dataFile.get(in);	
				inputLayer[k] = in;			
			}

			double* deltaCost1 = new double[10];
			
			//forward propagation
			for(int k = 0; k < 125; k++)
			{
				hiddenLayer[k] = sigmoid(matrixMultiplySum(inputLayer, weightsOne, k, 784) + hiddenLayerBias[k]);
			}
			for(int k = 0; k < 10; k++)
			{
				outputLayer[k] = sigmoid(matrixMultiplySum(hiddenLayer, weightsTwo, k, 125) + outputLayerBias[k]);
				if(outputLayer[k] > max) 
				{
					max = outputLayer[k];
					maxIndex = k;
				}
				deltaCost1[k] = 2.0 * (outputLayer[k] - expectedOutput[k]);
			}
			if(maxIndex == expectedindex)
			{
				score++;
			}
			max = 0;

			//std::cout << "j: " << j << std::endl;

			error += calcError(expectedOutput, outputLayer);
			//on to the back prop
			double* deltaCost = new double[125];
			deltaCost = backPropLayer(weightsTwo, outputLayerBias, deltaCost1, hiddenLayer, 125, 10, deltaCost, outputLayer);
			double* arbRet = new double[784];
			backPropLayer(weightsOne, hiddenLayerBias, deltaCost, inputLayer, 784, 125, arbRet, hiddenLayer);
			delete[] arbRet;
			delete[] deltaCost;				
			delete[] deltaCost1;
			delete[] expectedOutput;
		}
		std::cout << "Round " << i << " error: " << error/1001.0 << " : " << score/1001.0 << std::endl;
		for(int j = 0; j < 784; j++)
		{
			delete[] weightsOneDelta[j];
		}
		delete[] weightsOneDelta;
		for(int j = 0; j < 125; j++)
		{
			delete[] weightsTwoDelta[j];
		}
		delete[] weightsTwoDelta;
		delete[] hiddenBiasDelta;
		delete[] outputBiasDelta;

		//saving my neural network
		ofstream weightsOneOutputFile;
		weightsOneOutputFile.open("weightsOneOutputFile.txt");

		for(int i = 0; i < 784; i++)
		{
			for(int j = 0; j < 125; j++)
			{
				weightsOneOutputFile << weightsOne[i][j] << " ";
				weightsOneOutputFile << std::endl;
			}
			weightsOneOutputFile << std::endl;
		}
		
		ofstream hiddenLayerBiasOutputFile;
		hiddenLayerBiasOutputFile.open("hiddenLayerBiasOutputFile.txt");
		
		for(int i = 0; i < 125; i++)
		{
			hiddenLayerBiasOutputFile << hiddenLayerBias[i] << " ";
			hiddenLayerBiasOutputFile << std::endl;
		}

		ofstream weightsTwoOutputFile;
		weightsTwoOutputFile.open("weightsTwoOutputFile.txt");

		for(int i = 0; i < 125; i++)
		{
			for(int j = 0; j < 10; j++)
			{
				weightsTwoOutputFile << weightsTwo[i][j] << " ";
				weightsTwoOutputFile << std::endl;
			}
			weightsTwoOutputFile << std::endl;
		}

		ofstream outputLayerBiasOutputFile;
		outputLayerBiasOutputFile.open("outputLayerBiasOutputFile.txt");

		for(int i = 0; i < 10; i++)
		{
			outputLayerBiasOutputFile << outputLayerBias[i] << " ";
			outputLayerBiasOutputFile << std::endl;
		}
	}

	delete[] inputLayer;
	delete[] hiddenLayer;
	delete[] outputLayer;

	//defining biases of specific size
	delete[] hiddenLayerBias;
	delete[] outputLayerBias;
	for(int j = 0; j < 784; j++)
	{
		delete[] weightsOne[j];
	}
	delete[] weightsOne;
	for(int j = 0; j < 125; j++)
	{
		delete[] weightsTwo[j];
	}
	delete[] weightsTwo;
}


double matrixMultiplySum(double* inputs, double** weights, int indexInLayer, int weightsLen)
{
	double sum = 0.0;
	for(int i = 0; i < weightsLen; i++)
	{
		sum += inputs[i] * weights[i][indexInLayer];
	}
	return sum;
}

//Used for forward prop
double sigmoid(double x)
{
	return 1.0/(1.0 + pow(M_E,-x));
}

double* backPropLayer(double** weights, double* biases, double* deltaCost, double* prevActivation, int weightsLen, int weightsLenTwo, double* deltaNextActivation, double* nextActivation)
{
	double* deltaBiases = new double[weightsLenTwo];
	double** deltaWeights = new double* [weightsLen];
	for(int i = 0; i < weightsLen; i++)
	{
		deltaWeights[i] = new double [weightsLenTwo];
	}

	for(int j = 0; j < weightsLen; j++)
	{
		deltaNextActivation[j] = 0.0;
	}

	//setting the deltas
	for(int k = 0; k < weightsLen; k++)
	{
		for(int j = 0; j < weightsLenTwo; j++)
		{
			double sigmoidDerivativeResult = sigmoidDerivative(nextActivation[j]);
			
			deltaNextActivation[k] += weights[k][j] * sigmoidDerivativeResult * deltaCost[j];
			deltaWeights[k][j] = prevActivation[k] * sigmoidDerivativeResult * deltaCost[j];
			deltaBiases[j] = sigmoidDerivativeResult * deltaCost[j];
		}
	}

	//applying the deltas to the neural net
	for(int k = 0; k < weightsLen; k++)
	{
		for(int j = 0; j < weightsLenTwo; j++)
		{
			biases[j] -= .5 * deltaBiases[j];
			weights[k][j] -= .5 * deltaWeights[k][j];
		}
	}

	delete[] deltaBiases;
	for(int i = 0; i < weightsLen; i++)
	{
		delete[] deltaWeights[i];
	}
	delete[] deltaWeights;
	//return for next layer of backprop
	return deltaNextActivation;
}

//Used for back prop
double sigmoidDerivative(double x)
{
	return sigmoid(x) * (1.0 - sigmoid(x));
}