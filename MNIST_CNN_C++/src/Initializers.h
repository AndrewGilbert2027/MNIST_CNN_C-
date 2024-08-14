#pragma once
#include "Conv2D.h"

namespace setup {
	/// Randomly initializes a filter block 
	/// @param f The 2d filter size (f x f)
	/// @param n_c_prev The number of input channels
	/// @param n_c The number of output channels
	/// @return 4-dimensional randomly initialized weight of size (f x f x n_c_prev x n_c)
	dim4 initialize_random_weights(size_t f, size_t n_c_prev, size_t n_c);

	/// Randomly initializes biases for a convolutional block 
	/// @param n_c The number filter channels for the current convolutional block
	/// @return 4-dimensional array of size (1 x 1 x 1 x n_c) with random float values
	dim4 initialize_random_biases(size_t n_c);

	/// Randomly initializes a weight matrix for a fully connected layer
	/// @param output_dim The output dimension 
	/// @param input_dim The input dimension of the number of blocks
	/// @return 2-dimensional matrix of size (output_dim x input_dim)
	dim2 initialize_random_weights_dense(const size_t output_dim, const size_t input_dim);

	/// Randomly initializes a biases vector for a fully connected layer
	///	@param output_dim The output dimension vector.
	///	@return 1-dimensional float vector of size (output_dim)
	dim1 initialize_random_biases_dense(const size_t output_dim);
}