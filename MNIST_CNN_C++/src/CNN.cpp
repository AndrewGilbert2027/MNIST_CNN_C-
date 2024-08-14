#include "CNN.h"

#include <cassert>
#include <iostream>

#include "Backprop.h"
#include "initializers.h"
#include "Conv2D.h"
#include "DenseLayer.h"
#include "Pooling.h"


/*
 * Input dimensions (28 x 28 x 1)
 * First convolutional layer (5 x 5 x 1 x 10) (Padding=2, stride=2) -> (14 x 14 x 10)
 * Second convolutional layer (3 x 3 x 10 x 15) (Padding=0, stride = 2) -> (6 x 6 x 15)
 * Max pool layer (2 x 2) -> (5 x 5 x 15) (
 * Third convolutional layer (3 x 3 x 15 x 20) (Padding=2, stride = 1) -> (7 x 7 x 20)
 * Dense layer (flatten) (1 x 580) - > (1 x 10)
 */
CNN::CNN()
{
	//First convolutional layer weights
	W1 = setup::initialize_random_weights(5, 1, 10);
	b1 = setup::initialize_random_biases(10);

	//Second convolutional layer weights
	W2 = setup::initialize_random_weights(3, 10, 15);
	b2 = setup::initialize_random_biases(15);

	// Third convolutional layer
	W3 = setup::initialize_random_weights(3, 15, 20);
	b3 = setup::initialize_random_biases(20);

	// Dense layer convolutional weights
	W4 = setup::initialize_random_weights_dense(10, 980);
	b4 = setup::initialize_random_biases_dense(10);

	learning_rate_ = 0.0001;
}

dim2 CNN::forward(const dim4& A0)
{
	this->A0 = A0;
	// Implement first convolutional layer pass
	this->Z1 = convolutional::conv_forward(A0, W1, b1, 2, 2);
	this->A1 = convolutional::relu(Z1);

	// Implement second convolutional layer pass
	this->Z2 = convolutional::conv_forward(A1, W2, b2, 2, 0);
	this->A2 = convolutional::relu(Z2);

	// Max layer pooling
	this->A3 = pooling::pool_forward_max(A2, 2, 1);

	// Implement third convolutional layer pass
	this->Z4 = convolutional::conv_forward(A3, W3, b3, 1, 2);
	this->A4 = convolutional::relu(Z4);

	// Flatten output
	this->A5 = convolutional::flatten(A4);

	// Pass through linear full dense layer
	this->Z6 = dense_layer::forward_linear(A5, W4, b4);
	
	// Create softmax
	auto A6 = dense_layer::softmax(Z6);

	// Return m x y dimensional array where y is a vector of size 10 with class probabilities
	return A6;
}

void CNN::backpropagation(const dim2& output, const dim2& real)
{
    // Compute current dZ output - real
    const dim2 dZ_6 = backprop::subtract_matrix_2d(output, real);

    // Compute dW4 and db4
    const dim2 dW4 = dense_layer::backprop_dW(dZ_6, this->A5, W4);
    const dim1 db4 = dense_layer::backprop_db(dZ_6);
    assert(db4.size() == b4.size());

    // Compute dA5 (before flattening)
    const dim2 dA5 = dense_layer::backprop_dA(dZ_6, this->A5, W4);

    // Reshape dA5 from (580 x m) back to (m x 7 x 7 x 20)
    dim4 dA4 = convolutional::reshape_from_flatten(dA5, this->A4);

    // Compute dZ4 by applying ReLU derivative
    const dim4 dZ4 = convolutional::relu_derivative(dA4, this->Z4);

    // Compute dW3, db3, and dA3 for the third convolutional layer (padding=2, stride=1)
    constexpr size_t padding3 = 2;
    const dim4 dW3 = backprop::backprop_dW(this->A3, dZ4, 1, 3, padding3);
    const dim4 db3 = backprop::backprop_db(dZ4);
    const dim4 dA3 = backprop::backprop_dA(this->W3, dZ4, this->A3, 1, padding3);

    // Compute dA_pool using pooling layer's backpropagation
    const dim4 dA_pool = pooling::compute_dA_pool_max(this->A2, dA3, 1, 2);

    // Compute dZ2 by applying ReLU derivative
    const dim4 dZ2 = convolutional::relu_derivative(dA_pool, this->Z2);

    // Compute dW2, db2, and dA2 for the second convolutional layer (padding=0, stride=2)
    constexpr size_t padding2 = 0;
    const dim4 dW2 = backprop::backprop_dW(this->A1, dZ2, 2, 3, padding2);
    const dim4 db2 = backprop::backprop_db(dZ2);
    const dim4 dA2 = backprop::backprop_dA(this->W2, dZ2, this->A1, 2, padding2);

    // Compute dZ1 by applying ReLU derivative
    const dim4 dZ1 = convolutional::relu_derivative(dA2, this->Z1);

   
    // Compute dW1, db1, and dA1 for the first convolutional layer (padding=2, stride=2)
    constexpr size_t padding1 = 2;
    const dim4 dW1 = backprop::backprop_dW(A0, dZ1, 2, 5, padding1);
    const dim4 db1 = backprop::backprop_db(dZ1);

    // Update weights and biases with gradient descent
    W4 = backprop::update_weights_dense(W4, dW4, learning_rate_);
    b4 = backprop::update_biases_dense(b4, db4, learning_rate_);

    W3 = backprop::update_weights(W3, dW3, learning_rate_);
    b3 = backprop::update_biases(b3, db3, learning_rate_);

    W2 = backprop::update_weights(W2, dW2, learning_rate_);
    b2 = backprop::update_biases(b2, db2, learning_rate_);

    W1 = backprop::update_weights(W1, dW1, learning_rate_);
    b1 = backprop::update_biases(b1, db1, learning_rate_);
}

double compute_cross_entropy_loss_cnn(const dim2& output, const dim2& labels)
{
    const size_t classes = output.size();
    assert(classes == labels.size());
    const size_t m = output[0].size();
    assert(m == labels[0].size());

    double total_loss = 0.0;
    for (size_t i = 0; i < m; i++)
    {
        for (size_t c = 0; c < classes; c++)
        {
            total_loss -= labels[c][i] * std::log(output[c][i] + 1e-15);
        }
    }
    return total_loss / m; // Return average loss over the batch
}

void CNN::train(const dim4& input, const dim2& expected, const size_t epochs)
{
	for (size_t e = 0; e < epochs; e++)
	{
        auto predicted = this->forward(input);
        this->backpropagation(predicted, expected);
        auto loss = compute_cross_entropy_loss_cnn(predicted, expected);
        std::cout << "Current loss: " << loss << "\n";
	}
}

// Split training set into batches of size 30. 




