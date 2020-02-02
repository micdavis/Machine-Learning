main : Neuron.o main.o
	g++ Neuron.o main.o	

main.o : main.cc Neuron.h
	g++ -c main.cc

Neuron.o : Neuron.cc Neuron.h
	g++ -c Neuron.cc