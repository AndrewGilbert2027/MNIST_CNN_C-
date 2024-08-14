#include "Conv2D.h"

#include <cassert>
#include <iostream>


dim4 convolutional::zero_pad(const dim4& X, const size_t& pad)
{
	// Initialize output vector dimensions
	const size_t m = X.size();
	const size_t n_h = X[0].size();
	const size_t n_w = X[0][0].size();
	const size_t n_c = X[0][0][0].size();

	// Allocate memory for return vector
	dim4 X_pad = dim4(m, dim3(n_h + 2 * pad, dim2(n_w + 2 * pad, dim1(n_c, 0.0))));

	// Copy over values of input vector X to padded vector
	for (size_t i = 0; i < m; i++)
	{
		for (size_t k = 0; k < n_h; k++)
		{
			for (size_t l = 0; l < n_w; l++)
			{
				for (size_t c = 0; c < n_c; c++)
				{
					X_pad[i][k + static_cast<size_t>(pad)][l + static_cast<size_t>(pad)][c] = X[i][k][l][c];
				}
			}
		}
	}
	return X_pad;
}


dim3 convolutional::get_slice(const dim3& arr, const size_t& v_start, const size_t& v_end, const size_t& h_start, const size_t& h_end)
{
	// Retrieve output dimensions
	size_t v = v_end - v_start;
	size_t h = h_end - h_start;
	size_t n_c = arr[0][0].size();

	// Allocate memory for slice
	dim3 arr_slice = dim3(v, dim2(h, dim1(n_c, 0.0)));

	// Copy over values to output array
	for (size_t i = 0; i < v; i++)
	{
		for (size_t k = 0; k < h; k++)
		{
			for (size_t c = 0; c < n_c; c++)
			{
				assert(v_start + i < arr.size());
				assert(h_start + k < arr[0].size());
				arr_slice[i][k][c] = arr[v_start + i][h_start + k][c];
			}
		}
	}

	return arr_slice;
}

dim3 convolutional::get_filter_slice(const dim4& filter, const size_t& c)
{
	// Get output dimensions for current filter
	const size_t f = filter.size();
	const size_t n_c_prev = filter[0][0].size();

	// Allocate memory to copy over
	dim3 filter_slice = dim3(f, dim2(f, dim1(n_c_prev, 0.0)));

	// Copy desired filter contents over to current filter_slice
	for (size_t row = 0; row < f; row++)
	{
		for (size_t col = 0; col < f; col++)
		{
			for (size_t depth = 0; depth < n_c_prev; depth++)
			{
				filter_slice[row][col][depth] = filter[row][col][depth][c];
			}
		}
	}

	return filter_slice;
}

double convolutional::conv_single_step(const dim3& a_slice_prev, const dim3& W, double b)
{
	//Apply element wise multiplication between slice and filter
	const size_t f = W.size();
	const size_t n_c = W[0][0].size();
	double Z = 0.0;

	for (size_t r = 0; r < f; r++)
	{
		for (size_t c = 0; c < f; c++)
		{
			for (size_t d = 0; d < n_c; d++)
			{
				Z += a_slice_prev[r][c][d] * W[r][c][d];
			}
		}
	}

	Z += b;
	return Z;
}



dim4 convolutional::conv_forward(const dim4& A_prev, const dim4& W, const dim4& b, const size_t& stride, const size_t& pad)
{
	// Get dimensions of last convolutional block
	const size_t m = A_prev.size();
	const size_t n_h_prev = A_prev[0].size();
	const size_t n_w_prev = A_prev[0][0].size();
	const size_t n_c_prev = A_prev[0][0][0].size();

	// Retrieve dimensions of current filter block
	const size_t f = W.size();
	const size_t n_c = W[0][0][0].size();
	assert(n_c_prev == W[0][0].size());

	// Compute output dimensions of Convolutional output
	const size_t n_h = ((n_h_prev + (2 * pad) - f) / stride) + 1;
	const size_t n_w = ((n_w_prev + (2 * pad) - f) / stride) + 1;

	// Initialize output volume with zeros
	dim4 Z = dim4(m, dim3(n_h, dim2(n_w, dim1(n_c, 0.0))));

	// Pad A_prev
	const dim4 A_prev_pad = zero_pad(A_prev, pad);

	// Loop over batch of training examples
	for (size_t i = 0; i < m; i++)
	{
		const dim3& a_prev_pad = A_prev_pad[i];  // Select ith training example

		for (size_t v = 0; v < n_h; v++)  // loop over vertical axis of output volume
		{
			// Find vertical start and end of the current training 'slice'
			size_t v_start = v * stride;
			size_t v_end = v * stride + f;

			for (size_t h = 0; h < n_w; h++)  // Loop over the horizontal axis of output volume
			{
				// Find horizontal start and end of the current training 'slice'
				size_t h_start = h * stride;
				size_t h_end = h * stride + f;

				// Check slice bounds
				if (v_end > A_prev_pad[i].size() || h_end > A_prev_pad[i][0].size()) {
					std::cerr << "Slice out of bounds: "
						<< "v_start: " << v_start << ", v_end: " << v_end
						<< ", h_start: " << h_start << ", h_end: " << h_end << "\n";
					continue; // Skip this slice if it's out of bounds
				}

				for (size_t c = 0; c < n_c; c++)  // Loop over the number of channels (# of filters) of the output volume
				{
					// Get current slice to apply filter matrix to
					//std::cout << "Getting slice...\n";
					dim3 a_slice_prev = get_slice(a_prev_pad, v_start, v_end, h_start, h_end);

					//std::cout << "Getting filter slice...\n";
					dim3 filter_slice = get_filter_slice(W, c);

					// Get current biases to add to matrix
					const double bias = b[0][0][0][c];
					// Convolve the current 3D slice with the correct filter W and bias b
					Z[i][v][h][c] = conv_single_step(a_slice_prev, filter_slice, bias);

				}
			}
		}
	}

	return Z;
}

dim2 convolutional::flatten(const dim4& A)
{
	// Get dimensions of input image
	const size_t m = A.size();
	const size_t n_h = A[0].size();
	const size_t n_w = A[0][0].size();
	const size_t n_c = A[0][0][0].size();

	// Create output array of size (m x (n_h x n_w x n_c))
	dim2 output = dim2(n_h * n_w * n_c, dim1(m, 0.0));

	// Loop through input image starting
	for (size_t i = 0; i < m; i++)
	{
		for (size_t c = 0; c < n_c; c++)
		{
			for (size_t h = 0; h < n_h; h++)
			{
				for (size_t w = 0; w < n_w; w++)
				{
					// Calculate the correct index for the flattened array
					const size_t feature_i = c * n_h * n_w + h * n_w + w;
					output[feature_i][i] = A[i][h][w][c];
				}
			}
		}
	}
	return output;
}

dim4 convolutional::relu(const dim4& A)
{
	// Get dimensions of input vector
	const size_t m = A.size();
	const size_t n_h = A[0].size();
	const size_t n_w = A[0][0].size();
	const size_t n_c = A[0][0][0].size();

	// Initialize memory for output Array
	dim4 output = dim4(m, dim3(n_h, dim2(n_w, dim1(n_c, 0.0))));

	// Loop through input array and apply ReLu activation
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n_h; j++)
		{
			for (size_t k = 0; k < n_w; k++)
			{
				for (size_t c = 0; c < n_c; c++)
				{
					output[i][j][k][c] = (A[i][j][k][c] > 0) ? A[i][j][k][c] : 0;
				}
			}
		}
	}

	return output;
}


dim4 convolutional::relu_derivative(const dim4& input)
{
	// Get dimensions of input
	const size_t m = input.size();
	const size_t n_h = input[0].size();
	const size_t n_w = input[0][0].size();
	const size_t n_c = input[0][0][0].size();

	// Initialize memory for output array
	dim4 deriv = dim4(m, dim3(n_h, dim2(n_w, dim1(n_c, 0.0))));

	// Go through input
	for (size_t i = 0; i < m; i++)
	{
		for (size_t row = 0; row < n_h; row++)
		{
			for (size_t col = 0; col < n_w; col++)
			{
				for (size_t depth = 0; depth < n_c; depth++)
				{
					deriv[i][row][col][depth] = (input[i][row][col][depth] > 0.0) ? 1.0 : 0.0;
				}
			}
		}
	}

	return deriv;
}

dim4 convolutional::relu_derivative(const dim4& A, const dim4& Z)
{
	// Get dimensions of the input (Z)
	const size_t m = Z.size();
	const size_t n_h = Z[0].size();
	const size_t n_w = Z[0][0].size();
	const size_t n_c = Z[0][0][0].size();

	// Initialize output array for the derivatives
	dim4 dZ(m, dim3(n_h, dim2(n_w, dim1(n_c, 0.0))));

	// Iterate through each element in the input tensor (Z)
	for (size_t i = 0; i < m; i++)
	{
		for (size_t h = 0; h < n_h; h++)
		{
			for (size_t w = 0; w < n_w; w++)
			{
				for (size_t c = 0; c < n_c; c++)
				{
					// Apply the ReLU derivative: 1 if Z > 0, otherwise 0
					dZ[i][h][w][c] = (Z[i][h][w][c] > 0.0) ? A[i][h][w][c] : 0.0;
				}
			}
		}
	}

	return dZ;
}



dim4 convolutional::reshape_from_flatten(const dim2& dA5, const dim4& A4)
{
	// Get dimensions from the original 4D array
	const size_t m = A4.size();
	const size_t n_h = A4[0].size();
	const size_t n_w = A4[0][0].size();
	const size_t n_c = A4[0][0][0].size();

	// Initialize the output array with the same dimensions as A4
	dim4 dA4(m, dim3(n_h, dim2(n_w, dim1(n_c, 0.0))));

	// Loop through the flattened array and reshape it back into the 4D array
	for (size_t i = 0; i < m; i++)
	{
		for (size_t c = 0; c < n_c; c++)
		{
			for (size_t h = 0; h < n_h; h++)
			{
				for (size_t w = 0; w < n_w; w++)
				{
					// Calculate the correct index for the flattened array
					const size_t feature_i = c * n_h * n_w + h * n_w + w;
					dA4[i][h][w][c] = dA5[feature_i][i];
				}
			}
		}
	}

	return dA4;
}

dim3 convolutional::remove_padding(const dim3& input, const size_t padding)
{
	// Get dimensions of the input with padding
	const size_t n_h_padded = input.size();
	const size_t n_w_padded = input[0].size();
	const size_t n_c = input[0][0].size();

	// Calculate the dimensions of the output without padding
	const size_t n_h = n_h_padded - 2 * padding;
	const size_t n_w = n_w_padded - 2 * padding;

	// Initialize the output array without padding
	dim3 output(n_h, dim2(n_w, dim1(n_c, 0.0)));

	// Copy the values from the input tensor, excluding the padding
	for (size_t h = 0; h < n_h; h++)
	{
		for (size_t w = 0; w < n_w; w++)
		{
			for (size_t c = 0; c < n_c; c++)
			{
				output[h][w][c] = input[h + padding][w + padding][c];
			}
		}
	}

	return output;
}

dim4 convolutional::pad(const dim4& input, const size_t padding)
{
	// Get dimensions of the input without padding
	const size_t m = input.size();
	const size_t n_h = input[0].size();
	const size_t n_w = input[0][0].size();
	const size_t n_c = input[0][0][0].size();

	// Calculate the dimensions of the output with padding
	const size_t n_h_padded = n_h + 2 * padding;
	const size_t n_w_padded = n_w + 2 * padding;

	// Initialize the output array with padding, filled with zeros
	dim4 output(m, dim3(n_h_padded, dim2(n_w_padded, dim1(n_c, 0.0))));

	// Copy the values from the input tensor into the center of the padded output tensor
	for (size_t i = 0; i < m; i++)
	{
		for (size_t h = 0; h < n_h; h++)
		{
			for (size_t w = 0; w < n_w; w++)
			{
				for (size_t c = 0; c < n_c; c++)
				{
					output[i][h + padding][w + padding][c] = input[i][h][w][c];
				}
			}
		}
	}

	return output;
}







