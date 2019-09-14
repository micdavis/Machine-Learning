#include "Perceptron.h"
using namespace std;

class Perceptron
{
public:
	Perceptron::Perceptron(int numInputs, int numOutputs, float value)
	{
		input = new Weight[numInputs];
		output = new Weight[numOutputs];
		this->value = value;
	}
}