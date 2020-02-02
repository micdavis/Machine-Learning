#include <iostream>
#include "Neuron.h"

Neuron::Neuron()
{
  activation = 0;
  bias = 0;
  weights = 0;
  weightsLen = 0;
}

Neuron::Neuron(int prevLayerSize)
{
	activation = 0;
	bias = 0;

	if(prevLayerSize != 0) weights = new double[prevLayerSize];
	else weights = 0; 

  weightsLen = prevLayerSize;
}	

Neuron::Neuron(const Neuron &neuron)
{
  activation = neuron.activation;
  bias = neuron.bias;
  *weights = *neuron.weights;
  weightsLen = neuron.weightsLen;
}

double Neuron::getActivation()
{
  return activation;
}

bool Neuron::setActivation(double activation)
{
  this->activation = activation;
  return true;
}

double Neuron::getBias()
{
  return bias;
}

bool Neuron::setBias(double bias)
{
  this->bias = bias;
  return true;
}

double* Neuron::getWeights()
{
  return weights;
}

double Neuron::getWeight(int index)
{
  return weights[index];
}

bool Neuron::setWeights(double* weights, int weightsLen)
{
  this->weights = weights;
  this->weightsLen = weightsLen;
  return true;
}

bool Neuron::setWeight(int index, double weight)
{
  if(index < weightsLen) 
  {
    this->weights[index] = weight;
    return true;
  }
  else return false;
}

int Neuron::getWeightsLen()
{
  return this->weightsLen;
}

bool Neuron::printNeuron()
{
  std::cout.precision(2); 
  std::cout << "activation: " << activation << std::endl;
  std::cout << "bias: " << bias << std::endl;
  std::cout << "weights: ";
  for(int i = 0; i < weightsLen; i++)
  {
    if(i%10 == 0) std::cout << std::endl;
    std::cout << weights[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "weightsLen: " << weightsLen << std::endl;
  return true;
}