class Neuron
{
	public :
		double activation;
		double bias;
		double* weights;
		Neuron(int prevLayerSize);
};