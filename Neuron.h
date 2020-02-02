class Neuron
{
	private :
		double value;
		double delta;
		double bias;

	public:
		Neuron (double value);
		Neuron ();
		double getValue();
		double getDelta();
		double getBias();
		void setValue(double value);
		void setDelta(double delta);
		void setBias(double bias);
		void addDelta(double delta);
		void addBias(double bias);
};