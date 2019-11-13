#include "Matrix.h"

using namespace std;

template <typename T>

Matrix::Matrix(int size) {
	array = new int[size];
	this.size = size;
}

Matrix::Matrix(Matrix matrix) {
	array = matrix.array;
	size = martix.size;
}

Matrix::setMatrix(Matrix matrix) {
	this.array = matrix.array;
	this.size = matrix.size;
}

Matrix::getArray() {
	return array;
}

Matrix::getSize() {
	return size;
}

ostream& operator<<(ostream& os, const Matrix& matrix) {
	for(int i = 0; i < matrix.getSize(); i++)
	{
		os << matrix.getArray();
	}

	return os;
}

int main()
{
	Matrix matrix = new Matrix(5);
	cout << matrix;
	matrix.getArray()[3] = 4;
	cout << matrix; 
}