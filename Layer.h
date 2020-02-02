#include <array>
#include "Neuron.h"

class Layer
{
	public:
		std::array<Neuron> neurons;
		int size;
		Layer(int size);

}