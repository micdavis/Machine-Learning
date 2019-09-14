#include "Weight.h"
using namespace std;

class Weight
{
	Weight::Weight(Perceptron input, Perceptron output, float weight)
	{
		this->input = input;
		this->output = output;
		this->weight = weight; 
	}
}
