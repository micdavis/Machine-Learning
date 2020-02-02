#include "Neuron.h"

Neuron::Neuron(int prevLayerSize)
{
	activation = 0;
	bias = 0;

	if(prevLayerSize != 0) weights = new double[prevLayerSize];
	else weights = 0; 
}	

