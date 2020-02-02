#include "Neuron.h"
#include <iostream>
int main() {
	Neuron* n = new Neuron(5);
	std::cout << n->bias << std::endl;
}