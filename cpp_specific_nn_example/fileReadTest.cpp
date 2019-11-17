#include <iostream>
#include <fstream>
using namespace std;

int main()
{
	std::fstream labelFile("/home/michael/Documents/gitRepos/Machine-Learning/cpp_specific_nn_example/trainingData/train-labels-idx1-ubyte", std::ios_base::in);


	char c;
	while (labelFile >> c)
	{
		printf("%d\n", c);
	}

	std::cout << c << std::endl;
}