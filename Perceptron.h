#include "Weight.h"
using namspace std;
class Perceptron
{
public:
	float value;
	Weight *input;
	Weight *output;

	Perceptron (int, int, float);
};