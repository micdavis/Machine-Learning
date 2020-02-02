#include "Neuron.h"
#include <iostream>
int main() {
	Neuron* n = new Neuron(5);
  n->setBias(5.0);
  n->setActivation(18.0);
  double* weights = new double[5];
  for(int i = 0; i < 5; i++)
  {
    weights[i] = i;
  }
  n->setWeights(weights, 5);
  n->printNeuron();
  Neuron* m = n;
  m->printNeuron();
}