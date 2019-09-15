#include "Weight.h"
#include "Perceptron.h"
using namespace std;
class Network
{
	Perceptron *perceptron;
	Weight *weight;

	Network(int perceptronCountPerLayer, int numLayers);
}