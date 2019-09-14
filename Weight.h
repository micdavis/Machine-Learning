#include "Perceptron.h"
using namespace std;
class Weight
{
	float weight;
	Perceptron input;
	Perceptron output;

	Weight(Perceptron input, Perceptron output, float weight);
};