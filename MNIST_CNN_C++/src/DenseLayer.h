#pragma once
#include "Conv2D.h"
#include "cmath"
#include "algorithm"

namespace dense_layer
{
	/// Implement a forward linear pass in a fully connected layer.
	/// @param X the input arr of size (m x n_f) where m is the number of training examples and n_f is number of features
	/// @param W the weight matrix of size (output_d, input_d) where output_d is 10 and input_d is 1280 
	/// @param b the bias matrix of size (output_d)
	/// @return the linear forward pass with dimension (m x 10) for number of classes to recognize
	dim2 forward_linear(const dim2& X, const dim2& W, const dim1& b);

	/// Apply softmax function to rows in (m x 10) 2-dimensional vector
	/// @param X array of size (m x 10)
	/// @return softmax output of final probability scores for the mnist data set
	dim2 softmax(const dim2& X);

	dim2 backprop_dW(const dim2& dZ, const dim2& A_prev, const dim2& W);

	dim1 backprop_db(const dim2& dZ);

	dim2 backprop_dA(const dim2& dZ, const dim2& A_prev, const dim2& W);

	dim2 dot_product(const dim2& X, const dim2& Y);

	dim2 transpose(const dim2& Y);
}
