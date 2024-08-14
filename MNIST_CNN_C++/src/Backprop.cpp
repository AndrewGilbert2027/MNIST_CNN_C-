#include "Backprop.h"
#include <cassert>
#include <iostream>

dim2 backprop::back_prop_slice_dW(const dim2& a, const dim2& w, const dim2& dZ, const size_t stride)
{
	// Get dimensions and assert that everything lines up for debugging purposes
	const size_t n_h_prev = a.size();
	const size_t n_w_prev = a[0].size();

	const size_t f = w.size();
	assert(w[0].size() == f);

	const size_t n_h = dZ.size();
	const size_t n_w = dZ[0].size();

	assert(n_h == ((n_h_prev - f) / stride) + 1);
	assert(n_w == ((n_w_prev - f) / stride) + 1);

	// Initialize memory for derivative matrix for w
	dim2 dW = dim2(f, dim1(f, 0.0));

	// Loop through a and compute derivatives
	for (size_t height = 0; height < n_h; height++)
	{
		// Compute vertical start and end of current slice for a
		const size_t h_start = height * stride;
		for (size_t width = 0; width < n_w; width++)
		{
			// Compute horizontal start and end of current slice for a
			const size_t w_start = width * stride;

			// Loop through slice of dW and add matrices
			for (size_t i = 0; i < f; i++)
			{
				for (size_t k = 0; k < f; k++)
				{
					dW[i][k] += a[i + h_start][k + w_start] * dZ[height][width];
				}
			}

		}
	}

	return dW;
}

dim2 backprop::back_prop_slice_dA(const dim2& w, const dim2& a, const dim2& dZ, const size_t stride)
{
	const size_t n_h_prev = a.size();
	const size_t n_w_prev = a[0].size();

	const size_t f = w.size();
	assert(f == w[0].size());

	const size_t n_h = dZ.size();
	const size_t n_w = dZ[0].size();  

	assert(n_h == (n_h_prev - f) / stride + 1);
	assert(n_w == (n_w_prev - f) / stride + 1);

	dim2 dA = dim2(n_h_prev, dim1(n_w_prev, 0.0));

	for (size_t height = 0; height < n_h; height++)
	{
		const size_t h_start = height * stride;
		for (size_t width = 0; width < n_w; width++)
		{
			const size_t w_start = width * stride;

			for (size_t i = 0; i < f; i++)
			{
				for (size_t k = 0; k < f; k++)
				{
					dA[h_start + i][w_start + k] += w[i][k] * dZ[height][width];
				}
			}
		}
	}
	return dA;
}

dim3 backprop::back_prop_slice_dW_3d(const dim3& a_i, const dim2& dZ_i_c, const size_t stride, const size_t f)
{
	// Get dimensions arrays
	const size_t n_h_prev = a_i.size();
	const size_t n_w_prev = a_i[0].size();
	const size_t n_c_prev = a_i[0][0].size();

	const size_t n_h = dZ_i_c.size();
	const size_t n_w = dZ_i_c[0].size();
	assert(n_h == (n_h_prev - f) / stride + 1);
	assert(n_w == (n_w_prev - f) / stride + 1);

	// Initialize memory for output dW
	dim3 dW_c = dim3(f, dim2(f, dim1(n_c_prev, 0.)));

	// Loop through arrays and compute derivatives
	for (size_t z_row = 0; z_row < n_h; z_row++)
	{
		// Find vertical start coordinate of slice for a_i
		const size_t v_start = z_row * stride;
		for (size_t z_col = 0; z_col < n_w; z_col++)
		{
			// Find horizontal start coordinate of slice for a_i
			const size_t h_start = z_col * stride;
			for (size_t depth = 0; depth < n_c_prev; depth++)
			{
				// Loop through current channel of derivative matrix
				for (size_t i = 0; i < f; i++)
				{
					for (size_t k = 0; k < f; k++)
					{
						dW_c[i][k][depth] += a_i[v_start + i][h_start + k][depth] * dZ_i_c[z_row][z_col];
					}
				}
			}
			
		}
	}

	return dW_c;
}

dim3 backprop::back_prop_slice_dA_3d(const dim3& a_i, const dim3& w_c, const dim2& dZ, const size_t stride)
{
	// Get dimensions of arrays
	const size_t n_h_prev = a_i.size();
	const size_t n_w_prev = a_i[0].size();
	const size_t n_c_prev = a_i[0][0].size();

	const size_t f = w_c.size();
	assert(f == w_c[0].size());
	assert(n_c_prev == w_c[0][0].size());

	const size_t n_h = dZ.size();
	const size_t n_w = dZ[0].size();
	
	assert(n_h == (n_h_prev - f) / stride + 1);
	assert(n_w == (n_w_prev - f) / stride + 1);

	// Initialize memory for output array
	dim3 dA_i = dim3(n_h_prev, dim2(n_w_prev, dim1(n_c_prev, 0.0)));

	// Loop through each channel
	for (size_t depth = 0; depth < n_c_prev; depth++)
	{
		// Get current 2d slice of filter block to feed forward
		auto w_slice = get_2d_filter_slice(w_c, depth);

		// Get current 2d slice of input block to feed forward
		auto a_slice = get_2d_input_slice(a_i, depth);

		// Get 2d derivative matrix of the current slice with filter
		auto dA_i_d = back_prop_slice_dA(w_slice, a_slice, dZ, stride);

		// Copy over derivative values of array to dA_i
		move_derivative_values_at_depth(dA_i, dA_i_d, depth);
	}

	return dA_i;
}



dim4 backprop::backprop_dA(const dim4& W, const dim4& dZ, const dim4& A, const size_t stride)
{
	// Get dimensions of matrices
	const size_t f = W.size();
	assert(f == W[0].size());
	const size_t n_c_prev = W[0][0].size();
	const size_t n_c = W[0][0][0].size();

	const size_t m = A.size();
	const size_t n_h_prev = A[0].size();
	const size_t n_w_prev = A[0][0].size();
	assert(n_c_prev == A[0][0][0].size());

	assert(m == dZ.size());
	const size_t n_h = dZ[0].size();
	const size_t n_w = dZ[0][0].size();
	assert(n_c == dZ[0][0][0].size());

	// Initialize memory for output matrix
	dim4 dA(m, dim3(n_h_prev, dim2(n_w_prev, dim1(n_c_prev, 0.0))));  // Preallocation

	// Loop through training examples
	for (size_t i = 0; i < m; i++)
	{
		// Get dz for ith training example
		const dim3& dZ_i = dZ[i];

		// Get ith training example for A
		const dim3& A_i = A[i];

		// Loop through each output channel (n_c) and compute derivatives
		for (size_t c = 0; c < n_c; c++)
		{
			// Get 2d slice of derivative dZ_i for the current channel
			auto dZ_i_c = get_2d_Z_slice(dZ_i, c);

			// Get 3d cube of the cth convolutional block
			auto w_c = convolutional::get_filter_slice(W, c);

			// Compute 3d dA matrix for current training example and 2d slice of z
			auto dA_i_c = back_prop_slice_dA_3d(A_i, w_c, dZ_i_c, stride);

			// Add current 3d derivative block to dA_i
			add_matrix_3d(dA[i], dA_i_c);  // Adding directly to the preallocated dA
		}
	}
	return dA;
}


dim4 backprop::backprop_dW(const dim4& A, const dim4& dZ, const size_t stride, const size_t f)
{
	// Get dimensions of arrays
	const size_t m = A.size();
	const size_t n_h_prev = A[0].size();
	const size_t n_w_prev = A[0][0].size();
	const size_t n_c_prev = A[0][0][0].size();

	assert(m == dZ.size());
	const size_t n_h = dZ[0].size();
	const size_t n_w = dZ[0][0].size();
	const size_t n_c = dZ[0][0][0].size();
	assert(n_h == (n_h_prev - f) / stride + 1);
	assert(n_w == (n_w_prev - f) / stride + 1);

	// Create memory for output dimensions
	dim4 dW = dim4(f, dim3(f, dim2(n_c_prev, dim1(n_c))));

	// Loop through training examples
	for (size_t i = 0; i < m; i++)
	{
		// Select the ith training examples
		const dim3& a_i = A[i];
		const dim3& dz_i = dZ[i];

		// Loop through each channel of dz_i to compute cth filter for ith training example
		for (size_t c = 0; c < n_c; c++)
		{
			// Get dz_i slice for given filter
			const dim2 dz_i_c = get_2d_Z_slice(dz_i, c);

			// Compute 3-dimensional filter matrix for cth filter
			const dim3 dW_c = back_prop_slice_dW_3d(a_i, dz_i_c, stride, f);

			// Add current derivative matrix to dW
			add_Wc_derivatives(dW, dW_c, c);
		}
	}
	// divide the entire matrix by the number of training examples
	divide_array(dW, static_cast<float>(m));
	return dW;
}

dim4 backprop::backprop_db(const dim4& dZ)
{
	// Get dimensions
	const size_t m = dZ.size();
	const size_t n_h = dZ[0].size();
	const size_t n_w = dZ[0][0].size();
	const size_t n_c = dZ[0][0][0].size();

	// Initialize memory for db matrix
	dim4 db = dim4(1, dim3(1, dim2(1, dim1(n_c, 0.0))));

	// Loop through dZ and compute db
	for (size_t i = 0; i < m; i++)
	{
		for (size_t c = 0; c < n_c; c++)
		{
			for (size_t k = 0; k < n_h; k++)
			{
				for (size_t l = 0; l < n_w; l++)
				{
					db[0][0][0][c] += dZ[i][k][l][c];
				}
			}
		}
	}

	// Divide b by the number of training examples
	divide_array(db, static_cast<float>(m));
	return db;
}





dim2 backprop::get_2d_filter_slice(const dim3& w, const size_t n_c_prev)
{
	// Get dimensions
	const size_t f = w.size();
	assert(f == w[0].size());
	assert(n_c_prev < w[0][0].size());

	// Initialize memory for output
	dim2 w_slice = dim2(f, dim1(f, 0.0));

	// Copy over values from current slice
	for (size_t i = 0; i < f; i++)
	{
		for (size_t k = 0; k < f; k++)
		{
			w_slice[i][k] = w[i][k][n_c_prev];
		}
	}
	return w_slice;
}

dim2 backprop::get_2d_input_slice(const dim3& a, const size_t n_c_prev)
{
	// Get dimensions
	const size_t n_h_prev = a.size();
	const size_t n_w_prev = a[0].size();
	assert(n_c_prev < a[0][0].size());

	// Initialize memory for output
	dim2 a_slice = dim2(n_h_prev, dim1(n_w_prev, 0.0));

	// Copy over values from current feature channel
	for (size_t i = 0; i < n_h_prev; i++)
	{
		for (size_t k = 0; k < n_w_prev; k++)
		{
			a_slice[i][k] = a[i][k][n_c_prev];
		}
	}
	return a_slice;
}

dim2 backprop::get_2d_Z_slice(const dim3& dZ_i, const size_t n_c)
{
	// Get dimensions
	const size_t n_h = dZ_i.size();
	const size_t n_w = dZ_i[0].size();
	assert(n_c < dZ_i[0][0].size());

	// Initialize memory for output slice
	dim2 dZ_i_c = dim2(n_h, dim1(n_w, 0.0));

	// Copy over values at channel n_c and return slice
	for (size_t i = 0; i < n_h; i++)
	{
		for(size_t k = 0; k < n_w; k++)
		{
			dZ_i_c[i][k] = dZ_i[i][k][n_c];
		}
	}
	return dZ_i_c;
}


void backprop::move_derivative_values_at_depth(dim3& dA_i, dim2& dA_i_d, const size_t& d)
{
	// Get dimensions
	const size_t n_h_prev = dA_i.size();
	const size_t n_w_prev = dA_i[0].size();
	assert(d < dA_i[0][0].size());
	assert(dA_i_d.size() == n_h_prev);
	assert(dA_i_d[0].size() == n_w_prev);

	// Copy values over of 2d slice to the correct depth of dA_i
	for (size_t i = 0; i < n_h_prev; i++)
	{
		for (size_t k = 0; k < n_w_prev; k++)
		{
			dA_i[i][k][d] += dA_i_d[i][k];
		}
	}
}

void backprop::add_matrix_3d(dim3& anchor, const dim3& adder)
{
	// Get dimensions
	const size_t n_h_prev = anchor.size();
	const size_t n_w_prev = anchor[0].size();
	const size_t n_c_prev = anchor[0][0].size();
	assert(n_h_prev == adder.size());
	assert(n_w_prev == adder[0].size());
	assert(n_c_prev == adder[0][0].size());

	// Loop through arrays and add values
	for (size_t i = 0; i < n_h_prev; i++)
	{
		for (size_t k = 0; k < n_w_prev; k++)
		{
			for (size_t c = 0; c < n_c_prev; c++)
			{
				anchor[i][k][c] += adder[i][k][c];
			}
		}
	}
}

void backprop::add_matrix_4d(dim4& anchor, const dim4& adder)
{
	// Get dimensions
	const size_t i = anchor.size();
	const size_t j = anchor[0].size();
	const size_t k = anchor[0][0].size();
	const size_t l = anchor[0][0][0].size();
	assert(i == adder.size());
	assert(j == adder[0].size());
	assert(k == adder[0][0].size());
	assert(l == adder[0][0][0].size());

	// Loop through arrays and add to anchor
	for (size_t x = 0; x < i; x++)
	{
		for (size_t y = 0; y < j; y++)
		{
			for (size_t z = 0; z < k; z++)
			{
				for (size_t p = 0; p < l; p++)
				{
					anchor[x][y][z][p] += adder[x][y][z][p];
				}
			}
		}
	}
}

void backprop::add_Wc_derivatives(dim4& dW, const dim3& dW_i_c, const size_t c)
{
	// Get dimensions of arrays
	const size_t f = dW.size();
	assert(dW[0].size() == f);
	const size_t n_c_prev = dW[0][0].size();
	const size_t n_c = dW[0][0][0].size();
	assert(dW_i_c.size() == f);
	assert(dW_i_c.size() == dW_i_c[0].size());
	assert(dW_i_c[0][0].size() == n_c_prev);
	assert(c < n_c_prev);

	// Loop through dw_i_c and add to dW
	for (size_t i = 0; i < f; i++)
	{
		for (size_t j = 0; j < f; j++)
		{
			for (size_t l = 0; l < n_c; l++)
			{
				dW[i][j][l][c] += dW_i_c[i][j][l];
			}
		}
	}
}

void backprop::divide_array(dim4& dW, const float m)
{
	// Get dimensions of arrays
	const size_t f = dW.size();
	assert(dW[0].size() == f);
	const size_t n_c_prev = dW[0][0].size();
	const size_t n_c = dW[0][0][0].size();

	for (size_t i = 0; i < f; i++)
	{
		for (size_t k = 0; k < f; k++)
		{
			for (size_t j = 0; j < n_c_prev; j++)
			{
				for (size_t l = 0; l < n_c; l++)
				{
					dW[i][k][j][l] /= m;
				}
			}
		}
	}
}

dim2 backprop::subtract_matrix_2d(const dim2& anchor, const dim2& sub)
{
	// Get dimensions
	const size_t m = anchor.size();
	const size_t n = anchor[0].size();
	assert(m == sub.size());
	assert(n == sub[0].size());

	// Create output matrix
	dim2 output = dim2(m, dim1(n, 0.0));

	// subtract matrices into anchor
	for (size_t r = 0; r < m; r++)
	{
		for (size_t c = 0; c < n; c++)
		{
			output[r][c] = anchor[r][c] - sub[r][c];
		}
	}
	return output;
}

dim2 backprop::update_weights_dense(const dim2& W, const dim2& dW, const double& lr)
{
	// Get dimensions of weights
	const size_t rows = W.size();
	const size_t cols = W[0].size();
	assert(rows == dW.size());
	assert(cols == dW[0].size());

	// Initialize updated weights with the same dimensions as W
	dim2 updated_W = W;

	// Update weights using gradient descent
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			updated_W[i][j] -= lr * dW[i][j];
		}
	}

	return updated_W;
}


dim1 backprop::update_biases_dense(const dim1& b, const dim1& db, const double& lr)
{
	// Get the size of the bias vector
	const size_t size = b.size();
	assert(size == db.size());

	// Initialize updated biases with the same dimensions as b
	dim1 updated_b = b;

	// Update biases using gradient descent
	for (size_t i = 0; i < size; i++)
	{
		updated_b[i] -= lr * db[i];
	}

	return updated_b;
}

dim4 backprop::update_weights(const dim4& W, const dim4& dW, const double& lr)
{
	// Get dimensions of weights
	const size_t f = W.size();
	assert(f == W[0].size());
	const size_t n_c_prev = W[0][0].size();
	const size_t n_c = W[0][0][0].size();
	assert(dW.size() == f);
	assert(dW[0].size() == f);
	assert(dW[0][0].size() == n_c_prev);
	assert(dW[0][0][0].size() == n_c);

	// Initialize updated weights with the same dimensions as W
	dim4 updated_W = W;

	// Update weights using gradient descent
	for (size_t i = 0; i < f; i++)
	{
		for (size_t j = 0; j < f; j++)
		{
			for (size_t k = 0; k < n_c_prev; k++)
			{
				for (size_t l = 0; l < n_c; l++)
				{
					updated_W[i][j][k][l] -= lr * dW[i][j][k][l];
				}
			}
		}
	}

	return updated_W;
}


dim4 backprop::update_biases(const dim4& b, const dim4& db, const double& lr)
{
	// Get dimensions of biases
	const size_t m = b.size();
	const size_t n_h = b[0].size();
	const size_t n_w = b[0][0].size();
	const size_t n_c = b[0][0][0].size();
	assert(m == db.size());
	assert(n_h == db[0].size());
	assert(n_w == db[0][0].size());
	assert(n_c == db[0][0][0].size());

	// Initialize updated biases with the same dimensions as b
	dim4 updated_b = b;

	// Update biases using gradient descent
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n_h; j++)
		{
			for (size_t k = 0; k < n_w; k++)
			{
				for (size_t l = 0; l < n_c; l++)
				{
					updated_b[i][j][k][l] -= lr * db[i][j][k][l];
				}
			}
		}
	}

	return updated_b;
}


dim4 backprop::backprop_dW(const dim4& A, const dim4& dZ, const size_t stride, const size_t f, const size_t padding)
{
	// Get dimensions of arrays
	const size_t m = A.size();
	const size_t n_h_prev = A[0].size();
	const size_t n_w_prev = A[0][0].size();
	const size_t n_c_prev = A[0][0][0].size();

	const size_t n_h = dZ[0].size();
	const size_t n_w = dZ[0][0].size();
	const size_t n_c = dZ[0][0][0].size();

	// Create memory for output dimensions
	dim4 dW = dim4(f, dim3(f, dim2(n_c_prev, dim1(n_c, 0.0))));

	// Pad the input activations to account for padding
	dim4 A_padded = padding > 0 ? convolutional::pad(A, padding) : A;

	// Loop through training examples
	for (size_t i = 0; i < m; i++)
	{
		// Select the ith training example
		const dim3& a_i_padded = A_padded[i];
		const dim3& dz_i = dZ[i];

		// Loop through each channel of dz_i to compute cth filter for ith training example
		for (size_t c = 0; c < n_c; c++)
		{
			// Get dz_i slice for given filter
			const dim2 dz_i_c = get_2d_Z_slice(dz_i, c);

			// Loop over vertical and horizontal positions of the output volume
			for (size_t v = 0; v < n_h; v++)
			{
				size_t v_start = v * stride;
				size_t v_end = v_start + f;

				for (size_t h = 0; h < n_w; h++)
				{
					size_t h_start = h * stride;
					size_t h_end = h_start + f;

					// Ensure slices stay within bounds
					assert(v_end <= a_i_padded.size());
					assert(h_end <= a_i_padded[0].size());

					// Get current slice from the padded input
					dim3 a_slice = convolutional::get_slice(a_i_padded, v_start, v_end, h_start, h_end);

					// Update the gradient for the weights
					for (size_t d = 0; d < n_c_prev; d++)
					{
						for (size_t p = 0; p < f; p++)
						{
							for (size_t q = 0; q < f; q++)
							{
								dW[p][q][d][c] += a_slice[p][q][d] * dz_i_c[v][h];
							}
						}
					}
				}
			}
		}
	}

	// Divide the entire matrix by the number of training examples
	divide_array(dW, static_cast<float>(m));
	return dW;
}


dim4 backprop::backprop_dA(const dim4& W, const dim4& dZ, const dim4& A, const size_t stride, const size_t padding)
{
	// Get dimensions of matrices
	const size_t f = W.size();
	const size_t n_c_prev = W[0][0].size();
	const size_t n_c = W[0][0][0].size();

	const size_t m = A.size();
	const size_t n_h_prev = A[0].size();
	const size_t n_w_prev = A[0][0].size();

	const size_t n_h = dZ[0].size();
	const size_t n_w = dZ[0][0].size();

	// Initialize memory for output matrix
	dim4 dA(m, dim3(n_h_prev, dim2(n_w_prev, dim1(n_c_prev, 0.0))));  // Preallocation

	// Pad dA with zeros to account for padding
	dim4 dA_padded = padding > 0 ? convolutional::pad(dA, padding) : dA;

	// Loop through training examples
	for (size_t i = 0; i < m; i++)
	{
		// Get dz for ith training example
		const dim3& dZ_i = dZ[i];

		// Loop through each output channel (n_c) and compute derivatives
		for (size_t c = 0; c < n_c; c++)
		{
			// Get 2d slice of derivative dZ_i for the current channel
			auto dZ_i_c = get_2d_Z_slice(dZ_i, c);

			// Get 3d cube of the cth convolutional block
			auto w_c = convolutional::get_filter_slice(W, c);

			// Compute 3d dA matrix for current training example and 2d slice of z
			auto dA_i_c_padded = back_prop_slice_dA_3d(dA_padded[i], w_c, dZ_i_c, stride);

			// Add current 3d derivative block to dA_i (after removing padding)
			add_matrix_3d(dA[i], convolutional::remove_padding(dA_i_c_padded, padding));
		}
	}

	return dA;
}