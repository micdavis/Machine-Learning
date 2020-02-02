#include <array>
#include <memory>
#include "Layer.h"

Layer::Layer(int size)
{
	this->size = size;
  for(int i = 0; i < size; i++)
  {
    neuronList.push_back(new Neuron());
  }
}