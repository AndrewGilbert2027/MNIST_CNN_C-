#include "DenseLayer.h"
#include <cassert>
#include <iostream>

dim2 dense_layer::forward_linear(const dim2& X, const dim2& W, const dim1& b) {
    // Get dimensions of input and weights
    const size_t n_f = X.size();            // Number of input features
    const size_t m = X[0].size();           // Number of samples in batch


    const size_t n_classes = W.size();          // Number of output neurons
    const size_t w_n = W[0].size();             // Number of input features (should match `features`)

    assert(w_n == n_f);
    assert(n_classes == b.size());              // Bias should match number of output neurons

    // Initialize output dimensions
    dim2 output(n_classes, dim1(m, 0.0));

    // Loop through output
    for (size_t r = 0; r < n_classes; r++)
    {
	    for (size_t c = 0; c < m; c++)
	    {
		    // Loop through w_n to compute output
            for (size_t k = 0; k < w_n; k++)
            {
                output[r][c] += W[r][k] * X[k][c];
            }
            output[r][c] += b[r];
	    }
    }

    return output;
}

dim2 dense_layer::softmax(const dim2& X)
{
    // Get the number of classes and number of samples
    const size_t num_classes = X.size();    // Number of classes, should be 10 for MNIST
    const size_t m = X[0].size();           // Number of samples

    // Initialize the output array with the same dimensions as input
    dim2 output(num_classes, dim1(m, 0.0f));

    // Apply softmax to each sample (column)
    for (size_t i = 0; i < m; i++) {
        // Compute the maximum value in the column for numerical stability
        const double max_val = *std::max_element(X.begin(), X.end(),
            [i](const dim1& a, const dim1& b) { return a[i] < b[i]; })->begin() + i;

        // Compute the denominator of the softmax function (sum of exponentials)
        double sum_exp = 0.0;
        for (size_t j = 0; j < num_classes; j++) {
            sum_exp += std::exp(X[j][i] - max_val); // Use max_val to prevent overflow
        }

        // Compute the softmax output for each class
        for (size_t j = 0; j < num_classes; j++) {
            output[j][i] = std::exp(X[j][i] - max_val) / sum_exp;
        }
    }

    return output;
}

dim2 dense_layer::backprop_dA(const dim2& dZ, const dim2& A_prev, const dim2& W)
{
    // Get dimensions and make sure they align
    const size_t output_classes = W.size();
    const size_t num_features = W[0].size();
    assert(A_prev.size() == num_features);
    const size_t m = A_prev[0].size();
    assert(output_classes == dZ.size());
    assert(m == dZ[0].size());
    return dot_product(transpose(W), dZ);
}


dim1 dense_layer::backprop_db(const dim2& dZ)
{
	// Get dimensions
    const size_t m = dZ.size();
    const size_t n = dZ[0].size();

    // Get output dimensions
    dim1 db = dim1(m, 0.0);

    // Loop over rows and compute sum to find db
    for (size_t r = 0; r < m; r++)
    {
	    for (size_t c = 0; c < n; c++)
	    {
            db[r] += dZ[r][c];
	    }
    }

    return db;
}



dim2 dense_layer::backprop_dW(const dim2& dZ, const dim2& A_prev, const dim2& W)
{
	// Get dimensions of arrays
    const size_t z_rows = dZ.size();
    const size_t z_cols = dZ[0].size();

    const size_t n_features = A_prev.size();
    const size_t m = A_prev[0].size();

    const size_t w_output = W.size();
    const size_t w_input = W[0].size();

    assert(w_input == n_features);
    assert(z_rows == w_output);
    assert(z_cols == m);

	// Compute derivative
    auto dW = dot_product(dZ, transpose(A_prev));
    return dW;
}

dim2 dense_layer::dot_product(const dim2& X, const dim2& Y)
{
	// Get dimensions
    const size_t x_m = X.size();
    const size_t x_n = X[0].size();
    const size_t y_m = Y.size();
    const size_t y_n = Y[0].size();
    assert(x_n == y_m);

    // Create output dimensions
    dim2 product = dim2(x_m, dim1(y_n, 0.0));

    // Loop through product and compute dot_product
    for (size_t r = 0; r < x_m; r++)
    {
	    for (size_t c = 0; c < y_n; c++)
	    {
		    for (size_t k = 0; k < x_n; k++)
		    {
                product[r][c] += X[r][k] * Y[k][c];
		    }
	    }
    }

    return product;
}


dim2 dense_layer::transpose(const dim2& Y)
{

	// Get dimensions of array
    const size_t m = Y.size();
    const size_t n = Y[0].size();
    dim2 transpose = dim2(n, dim1(m, 0.0));

    // Loop through Y and copy over values to return
    for (size_t i = 0; i < m; i++)
    {
	    for (size_t k = 0; k < n; k++)
	    {
            transpose[k][i] = Y[i][k];
	    }
    }
    return transpose;
}






