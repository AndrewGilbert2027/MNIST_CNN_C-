#pragma once
#include "Conv2D.h"

class CNN
{
public:
	CNN();

	dim2 forward(const dim4& A0);
	void backpropagation(const dim2& output, const dim2& real);
	void train(const dim4& input, const dim2& expected, const size_t epochs);

private:
	//
	double learning_rate_;
	// Weights and parameters
	dim4 W1;
	dim4 b1;

	dim4 W2;
	dim4 b2;

	dim4 W3;
	dim4 b3;

	// Fully connected layer
	dim2 W4;
	dim1 b4;

	dim4 A0;
	// Implement some sort of cache to easily use backpropagation
	dim4 Z1;
	dim4 A1;

	dim4 Z2;
	dim4 A2;

	dim4 A3; // Max - pooling array

	dim4 Z4;
	dim4 A4;
	//    ^
	//    |
	//    |
	dim2 A5; // Flattened array from A4

	dim2 Z6;
};

