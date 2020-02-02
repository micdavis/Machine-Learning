#include <iostream>

int main()
{

  double* pointer;

  std::cout << sizeof(pointer) << std::endl;
  std::cout << sizeof(&pointer) << std::endl;
  std::cout << sizeof(*pointer) << std::endl;

  pointer = new double[5];

  std::cout << sizeof(pointer) << std::endl;
  std::cout << sizeof(&pointer) << std::endl;
  std::cout << sizeof(*pointer) << std::endl;
}
