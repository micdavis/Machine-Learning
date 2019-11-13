using namespace std;

template <typename T>
class Matrix {
	T* array;
	int size;

	Matrix(int size);
	
	Matrix(Matrix matrix);

	void setMatrix(Matrix matrix);
	
	T getArray();
	
	int getSize();

    friend ostream& operator<<(ostream& os, const Matirx& matrix);
}

