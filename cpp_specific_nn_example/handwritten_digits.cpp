#include <cmath>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <cctype>
#include <unistd.h>
#include <time.h>
#include <thread>

#define HIDDENLAYERSIZE 500
#define INPUTLAYERSIZE 784
#define OUTPUTLAYERSIZE 10
#define NUMBEROFEPOCH 20
#define BATCHSIZE 10000

using namespace std;

double* backPropLayer(double** weights, double* biases, double* deltaCost, double* prevActivation, int weightsLen, int weigthLenTwo, double* deltaNextActivation, double* nextActivation);
double sigmoid(double x);
double sigmoidDerivative(double x);
double matrixMultiplySum(double* inputs, double** weights, int indexInLayer, int weightsLen);


double calcError(double* expected, double* actual)
{
	double sum = 0.0;
	for(int i = 0; i < OUTPUTLAYERSIZE; i++)
	{
		sum += (actual[i] - expected[i]) * (actual[i] - expected[i]);
	}
	return sum / (double)OUTPUTLAYERSIZE;
}


void loadData(double** inputData, double* inputLabels)
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

	for(int i = 0; i < BATCHSIZE; i++)
	{
		char in;
		labelFile.get(in);
		inputLabels[i] = (double) in;

		for(int k = 0; k < INPUTLAYERSIZE; k++)
		{
			char in;
			dataFile.get(in);	
			inputData[i][k] = (double)in;			
		}
	}
}

int main () {
	srand(time(NULL));

	//defining the input data storage
	double** inputData = new double* [BATCHSIZE];

	for(int i = 0; i < BATCHSIZE; i++)
	{
		inputData[i] = new double[INPUTLAYERSIZE];
	}

	double* inputLabels = new double[BATCHSIZE];

	loadData(inputData, inputLabels);

	//defining layers of specific sizeof
	double* inputLayer = new double[INPUTLAYERSIZE];
	double* hiddenLayer = new double[HIDDENLAYERSIZE];
	double* outputLayer = new double[OUTPUTLAYERSIZE];

	//defining biases of specific size
	double* hiddenLayerBias = new double[HIDDENLAYERSIZE];
	double* outputLayerBias = new double[OUTPUTLAYERSIZE];

	double max = 10.0;
	int maxIndex = 0;
	int expectedindex = 0;
	//defining weights of specific size
	double** weightsOne = new double* [INPUTLAYERSIZE];
	for(int i = 0; i < INPUTLAYERSIZE; i++)
	{
		weightsOne[i] = new double[HIDDENLAYERSIZE];
		for(int j = 0; j < HIDDENLAYERSIZE; j++)
		{
			weightsOne[i][j] = (double)(rand() % 20 - OUTPUTLAYERSIZE);
		}
	}

	double** weightsTwo = new double* [HIDDENLAYERSIZE];
	for(int i = 0; i < HIDDENLAYERSIZE; i++)
	{
		weightsTwo[i] = new double[OUTPUTLAYERSIZE];
		for(int j = 0; j < OUTPUTLAYERSIZE; j++)
		{
			weightsTwo[i][j] = (double)(rand() % 20 - OUTPUTLAYERSIZE);
		}
	}



	//number of learning cycles
	for(int i = 0; i < NUMBEROFEPOCH; i++)
	{
		double** weightsOneDelta = new double* [INPUTLAYERSIZE];
		for(int j = 0; j < INPUTLAYERSIZE; j++)
		{
			weightsOneDelta[j] = new double[HIDDENLAYERSIZE];
		}

		double* hiddenBiasDelta = new double [HIDDENLAYERSIZE];
		for(int q = 0; q < HIDDENLAYERSIZE; q++)
		{
			hiddenBiasDelta[q] = 0;
		}

		double** weightsTwoDelta = new double* [HIDDENLAYERSIZE]; 
		for(int j = 0; j < HIDDENLAYERSIZE; j++)
		{
			weightsTwoDelta[j] = new double[OUTPUTLAYERSIZE];
		}

		double* outputBiasDelta = new double[OUTPUTLAYERSIZE];
		for(int q = 0; q < OUTPUTLAYERSIZE; q++)
		{
			outputBiasDelta[q] = 0;
		}
		double error = 0.0;

		double score = 0.0;

		//number of test cases ran through
		for(int j = 0; j < BATCHSIZE; j++)
		{
			//set input layers to an input and expected output
			double* expectedOutput = new double[OUTPUTLAYERSIZE];
			for(int q = 0; q < OUTPUTLAYERSIZE; q++)
			{
				expectedOutput[q] = 0;
			}

			expectedOutput[(int)inputLabels[j]] = 1.0;

			expectedindex = inputLabels[j];

			inputLayer = inputData[j];

			double* deltaCost1 = new double[OUTPUTLAYERSIZE];
			
			//forward propagation
			for(int k = 0; k < HIDDENLAYERSIZE; k++)
			{
				hiddenLayer[k] = sigmoid(matrixMultiplySum(inputLayer, weightsOne, k, INPUTLAYERSIZE) + hiddenLayerBias[k]);
			}
			for(int k = 0; k < OUTPUTLAYERSIZE; k++)
			{
				outputLayer[k] = sigmoid(matrixMultiplySum(hiddenLayer, weightsTwo, k, HIDDENLAYERSIZE) + outputLayerBias[k]);
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
			expectedindex = 0;

			/*if(j % 500 == 0)
			{
				std::cout << "j: " << j << std::endl;				
			}*/

			error += calcError(expectedOutput, outputLayer);
			//on to the back prop
			double* deltaCost = new double[HIDDENLAYERSIZE];
			deltaCost = backPropLayer(weightsTwo, outputLayerBias, deltaCost1, hiddenLayer, HIDDENLAYERSIZE, OUTPUTLAYERSIZE, deltaCost, outputLayer);
			double* arbRet = new double[INPUTLAYERSIZE];
			backPropLayer(weightsOne, hiddenLayerBias, deltaCost, inputLayer, INPUTLAYERSIZE, HIDDENLAYERSIZE, arbRet, hiddenLayer);
			delete[] arbRet;
			delete[] deltaCost;				
			delete[] deltaCost1;
			delete[] expectedOutput;
		}
		std::cout << "Round " << i << " error: " << error/(double)BATCHSIZE << " : " << score/(double)BATCHSIZE << std::endl;
		for(int j = 0; j < INPUTLAYERSIZE; j++)
		{
			delete[] weightsOneDelta[j];
		}
		delete[] weightsOneDelta;
		for(int j = 0; j < HIDDENLAYERSIZE; j++)
		{
			delete[] weightsTwoDelta[j];
		}
		delete[] weightsTwoDelta;
		delete[] hiddenBiasDelta;
		delete[] outputBiasDelta;
	}
	delete[] inputLayer;
	delete[] hiddenLayer;
	delete[] outputLayer;

	//defining biases of specific size
	delete[] hiddenLayerBias;
	delete[] outputLayerBias;
	for(int j = 0; j < INPUTLAYERSIZE; j++)
	{
		delete[] weightsOne[j];
	}
	delete[] weightsOne;
	for(int j = 0; j < HIDDENLAYERSIZE; j++)
	{
		delete[] weightsTwo[j];
	}
	delete[] weightsTwo;

	//saving my neural network
	ofstream weightsOneOutputFile;
	weightsOneOutputFile.open("weightsOneOutputFile.txt");

	for(int i = 0; i < INPUTLAYERSIZE; i++)
	{
		for(int j = 0; j < HIDDENLAYERSIZE; j++)
		{
			weightsOneOutputFile << weightsOne[i][j] << " ";
			weightsOneOutputFile << std::endl;
		}
		weightsOneOutputFile << std::endl;
	}
	
	ofstream hiddenLayerBiasOutputFile;
	hiddenLayerBiasOutputFile.open("hiddenLayerBiasOutputFile.txt");
	
	for(int i = 0; i < HIDDENLAYERSIZE; i++)
	{
		hiddenLayerBiasOutputFile << hiddenLayerBias[i] << " ";
		hiddenLayerBiasOutputFile << std::endl;
	}

	ofstream weightsTwoOutputFile;
	weightsTwoOutputFile.open("weightsTwoOutputFile.txt");

	for(int i = 0; i < HIDDENLAYERSIZE; i++)
	{
		for(int j = 0; j < OUTPUTLAYERSIZE; j++)
		{
			weightsTwoOutputFile << weightsTwo[i][j] << " ";
			weightsTwoOutputFile << std::endl;
		}
		weightsTwoOutputFile << std::endl;
	}

	ofstream outputLayerBiasOutputFile;
	outputLayerBiasOutputFile.open("outputLayerBiasOutputFile.txt");

	for(int i = 0; i < OUTPUTLAYERSIZE; i++)
	{
		outputLayerBiasOutputFile << outputLayerBias[i] << " ";
		outputLayerBiasOutputFile << std::endl;
	}
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
			biases[j] -= .1 * deltaBiases[j];
			weights[k][j] -= .1 * deltaWeights[k][j];
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