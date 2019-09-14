CXX = g++
CXXFLAGS = -Wall -g

main: main.o Weight.o Network.o Perceptron.o
	$(CXX) $(CXXFLAGS) -o main main.o Weight.o Network.o Perceptron.o

