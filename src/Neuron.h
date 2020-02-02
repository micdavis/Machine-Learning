class Neuron
{
	private :
		double activation;
		double bias;
		double* weights;
    int weightsLen;

  public:
    Neuron();
		Neuron(int prevLayerSize);
    Neuron(const Neuron &neuron);
    double getActivation();
    bool setActivation(double activation);
    double getBias();
    bool setBias(double bias);
    double* getWeights();
    double getWeight(int index);
    bool setWeights(double* weights, int weightsLen);
    bool setWeight(int index, double weight);
    int getWeightsLen();
    bool printNeuron();
};