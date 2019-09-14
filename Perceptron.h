#include "Weight.h"
using namespace std;
class Perceptron
{
public:
	float value;
	Weight *input;
	Weight *output;

	Perceptron (int numInput, int numOutput, float value);
};