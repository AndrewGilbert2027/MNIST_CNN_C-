#include "initializers.h"
#include <random>

dim4 setup::initialize_random_weights(const size_t f, const size_t n_c_prev, const size_t n_c)
{
	// Seed random number generator
	std::random_device rd;
	std::mt19937 gen(rd());

	// Define uniform real distribution between -1 and 1
	std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

	// Allocate memory for random weight
	dim4 arr = dim4(f, dim3(f, dim2(n_c_prev, dim1(n_c, 0.0))));

	// Loop through new array and randomly generate numbers between -1 and 1
	for (size_t i = 0; i < f; i++)
	{
		for (size_t k = 0; k < f; k++)
		{
			for (size_t j = 0; j < n_c_prev; j++)
			{
				for (size_t m = 0; m < n_c; m++)
				{
					arr[i][k][j][m] = distribution(gen);
				}
			}
		}
	}
	return arr;
}

dim2 setup::initialize_random_weights_dense(const size_t output_dim, const size_t input_dim)
{
	// Seed random number generator
	std::random_device rd;
	std::mt19937 gen(rd());

	// Define uniform real distribution between -1 and 1
	std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

	// Create memory for output vector
	dim2 output = dim2(output_dim, dim1(input_dim, 0.0));

	// Create random weights
	for (size_t o = 0; o < output_dim; o++)
	{
		for (size_t i = 0; i < input_dim; i++)
		{
			output[o][i] = distribution(gen);
		}
	}
	return output;
}

dim1 setup::initialize_random_biases_dense(const size_t output_dim)
{
	// Seed random number generator
	std::random_device rd;
	std::mt19937 gen(rd());

	// Define uniform real distribution between -1 and 1
	std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

	// Create memory for output
	dim1 output = dim1(output_dim, 0.0);

	// Loop through output and return random biases
	for (size_t i = 0; i < output_dim; i++)
	{
		output[i] = distribution(gen);
	}
	return output;
}



dim4 setup::initialize_random_biases(const size_t n_c)
{
	// Seed random number generator
	std::random_device rd;
	std::mt19937 gen(rd());

	// Define uniform real distribution between -1 and 1
	std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

	// Allocate memory for random weight
	dim4 arr = dim4(1, dim3(1, dim2(1, dim1(n_c, 0.0))));

	// loop through array and add random biases
	for (size_t c = 0; c < n_c; c++)
	{
		arr[0][0][0][c] = distribution(gen);
	}

	return arr;
}

