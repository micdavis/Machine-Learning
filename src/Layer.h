#include <array>
#include <vector>
#include "Neuron.h"

class Layer
{
	private:
		std::vector<Neuron*> neuronList;
		int size;
  public:
		Layer(int size);
};