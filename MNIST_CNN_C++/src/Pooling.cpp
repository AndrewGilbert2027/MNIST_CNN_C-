#include "Pooling.h"

#include <cassert>
#include <iostream>

// Perform max pooling on a 4D input tensor
dim4 pooling::pool_forward_max(const dim4& A_prev, const size_t& f, const size_t& stride)
{
    // Retrieve the dimension sizes of the A_prev
    size_t m = A_prev.size();
    size_t n_h_prev = A_prev[0].size();
    size_t n_w_prev = A_prev[0][0].size();
    size_t n_c = A_prev[0][0][0].size();

    // Initialize output dimensions
    size_t n_h = (n_h_prev - f) / stride + 1;
    size_t n_w = (n_w_prev - f) / stride + 1;

    // Initialize output matrix
    dim4 A = dim4(m, dim3(n_h, dim2(n_w, dim1(n_c, 0.0))));

    for (size_t i = 0; i < m; i++) // Loop over the training examples
    {
        const dim3& a_prev = A_prev[i]; // Access ith training example
        for (size_t v = 0; v < n_h; v++) // Loop over vertical column 
        {
            // Get starting and ending coordinates of current slice
            size_t v_start = v * stride;
            size_t v_end = v * stride + f;

            for (size_t h = 0; h < n_w; h++) // Loop over horizontal axis 
            {
                // Get starting and ending horizontal coordinates of current slice
                size_t h_start = h * stride;
                size_t h_end = h * stride + f;

                for (size_t c = 0; c < n_c; c++) // Loop over the channels in output array
                {
                    // Get current slice
                    dim2 a_prev_slice = get_slice_pooling(a_prev, v_start, v_end, h_start, h_end, c);

                    // Assign max value to output
                    A[i][v][h][c] = get_max_in_slice(a_prev_slice);
                }
            }
        }
    }

    // Assert output shape is correct
    assert(A.size() == m);
    assert(A[0].size() == n_h);
    assert(A[0][0].size() == n_w);
    assert(A[0][0][0].size() == n_c);
    return A;
}

// Extract a slice from a 3D array for pooling
dim2 pooling::get_slice_pooling(const dim3& arr, const size_t& v_start, const size_t& v_end, const size_t& h_start, const size_t& h_end, const size_t& c)
{
    // Initialize memory for output array
    const size_t o_h = v_end - v_start;
    const size_t o_w = h_end - h_start;
    dim2 output = dim2(o_h, dim1(o_w, 0.0));

    // Loop over input array and assign values
    for (size_t r = 0; r < o_h; r++)
    {
        for (size_t w = 0; w < o_w; w++)  // Use distinct loop variable `w`
        {
            output[r][w] = arr[r + v_start][w + h_start][c]; // Use channel c directly
        }
    }

    return output;
}

// Find the maximum value in a 2D array
double pooling::get_max_in_slice(const dim2& arr)
{
    // Retrieve dimensions
    size_t v = arr.size();
    size_t h = arr[0].size();

    double max = arr[0][0];

    for (size_t i = 0; i < v; i++)
    {
        for (size_t k = 0; k < h; k++)
        {
            max = (max < arr[i][k]) ? arr[i][k] : max;
        }
    }
    return max;
}

pooling::coords pooling::get_max_coords(const dim2& array)
{
    // Get dimensions of array
    const size_t height = array.size();
    const size_t width = array[0].size();

    // Create memory to store x and y coord of max element
    size_t x_max = 0;
    size_t y_max = 0;
    double max_value = array[0][0];

    // Loop through array
    for (size_t i = 0; i < height; i++)
    {
	    for (size_t k = 0; k < width; k++)
	    {
		    if (array[i][k] > max_value)
		    {
                y_max = i;
                x_max = k;
                max_value = array[i][k];
		    }
	    }
    }
    return { x_max, y_max };
}


dim4 pooling::compute_dA_pool_max(const dim4& A, const dim4& dA_pool, const size_t stride, const size_t f)
{
    // Get dimensions of the input array
    const size_t m = A.size();
    const size_t n_h_prev = A[0].size();
    const size_t n_w_prev = A[0][0].size();
    const size_t n_c_prev = A[0][0][0].size();

    // Initialize the output derivative array (same size as A) with zeros
    dim4 dA_prev = dim4(m, dim3(n_h_prev, dim2(n_w_prev, dim1(n_c_prev, 0.0))));

    // Compute output dimensions
    const size_t n_h = (n_h_prev - f) / stride + 1;
    const size_t n_w = (n_w_prev - f) / stride + 1;

    // Check dimension consistency with dA_pool
    assert(m == dA_pool.size());
    assert(n_h == dA_pool[0].size());
    assert(n_w == dA_pool[0][0].size());
    assert(n_c_prev == dA_pool[0][0][0].size());

    // Loop over each example in the batch
    for (size_t i = 0; i < m; i++)
    {
        for (size_t c = 0; c < n_c_prev; c++)
        {
            for (size_t k = 0; k < n_h; k++)
            {
                for (size_t l = 0; l < n_w; l++)
                {
                    // Define the (v_start, h_start) of the current slice
                    const size_t v_start = k * stride;
                    const size_t h_start = l * stride;
                    const size_t v_end = v_start + f;
                    const size_t h_end = h_start + f;

                    // Ensure the slice doesn't go out of bounds
                    assert(v_end <= n_h_prev && h_end <= n_w_prev);

                    // Initialize variables to find the max value and its position
                    size_t x_max = v_start;
                    size_t y_max = h_start;
                    double max_value = A[i][v_start][h_start][c];

                    // Find the max value in the slice
                    for (size_t y = v_start; y < v_end; y++)
                    {
                        for (size_t x = h_start; x < h_end; x++)
                        {
                            if (A[i][y][x][c] > max_value)
                            {
                                max_value = A[i][y][x][c];
                                y_max = y;
                                x_max = x;
                            }
                        }
                    }

                    // Propagate the gradient only to the max value position
                    dA_prev[i][y_max][x_max][c] += dA_pool[i][k][l][c];
                }
            }
        }
    }

    return dA_prev;
}



