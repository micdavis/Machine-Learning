#include "Neuron.h"

int main() {
	Neuron n = new Neuron(5.0);
	std::cout << n.getBias();
}