#include "Neuron.h"

Neuron::Neuron(double bias)
	: bias(bias), value(0), delta(0)
{}

Neuron::Neuron()
	: bias(0), value(0), delta(0)
{}

double Neuron::getBias()
{
	return bias;
}