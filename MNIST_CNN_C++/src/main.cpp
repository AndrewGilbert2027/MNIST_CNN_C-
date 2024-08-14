#include <cassert>
#include <iostream>
#include "MnistLoader.h"
#include <vector>
#include <string>
#include "CNN.h"
#include "Conv2D.h"

dim4 reshape_array(const std::vector<std::vector<unsigned char>>& input, const size_t height, const size_t width);
void print_array(const dim2& arr);
double compute_cross_entropy_loss(const dim2& output, const dim2& labels);
dim2 reshape_labels_array(const std::vector<unsigned char>& labels, size_t classes);

int main(void)
{
    // String paths for pixel maps and labels of mnist data set
	std::string images_path = "C:/Users/d81ru/OneDrive/Desktop/c++DEV/MNIST_CNN_C++/MNIST_CNN_C++/MNIST_data/train-images.idx3-ubyte";
	std::string labels_path = "C:/Users/d81ru/OneDrive/Desktop/c++DEV/MNIST_CNN_C++/MNIST_CNN_C++/MNIST_data/train-labels.idx1-ubyte";

	// Initialize buffers to store Mnist training data
	std::vector<std::vector<unsigned char>> images;
	std::vector<unsigned char> labels;

	// Load Mnist training data into buffers by passing reference
    try {
        std::cout << "Loading dataset...\n";
        MNISTLoader::load_mnist(images_path, labels_path, images, labels);
        std::cout << "Done loading dataset!\n";

        std::cout << "Current number of training examples: " << images.size() << "\n";
        std::cout << "First label: " << static_cast<int>(labels[0]) << "\n";
        std::cout << "First image pixels:" << "\n";

        for (int y = 0; y < MNISTLoader::IMAGE_HEIGHT; ++y) {
            for (int x = 0; x < MNISTLoader::IMAGE_WIDTH; ++x) {
                std::cout << (images[0][y * MNISTLoader::IMAGE_WIDTH + x] > 60 ? '#' : ' ');
            }
            std::cout << "\n";
        }
        std::cout << "Done drawing!\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    // Convert training set to float precision before training model and also vector of size (m, n_h, n_w, 1) and divide by 255
    std::cout << "Reshaping array...\n";
    dim4 X = reshape_array(images, 28, 28);
    dim2 Y = reshape_labels_array(labels, 10);
    assert(X[0][0][0].size() == 1); // Grayscale
    std::cout << "Done reshaping array\n";

    std::cout << "Initializing current model...\n";
    std::string optimizer = "adam";
    CNN model;
    std::cout << "Done initializing model!\n";

    std::cout << "Passing input into model...\n";
    model.train(X, Y, 100);

    // Test first pass on one input
    // model.train(images, labels, optimizer);
    // model.show_results();
    // model.save(output-file);

    std::cin.get();
    return 0;
}

dim4 reshape_array(const std::vector<std::vector<unsigned char>>& input, const size_t height, const size_t width)
{
	// Get dimensions of input
    size_t m = input.size();
    size_t f = input[0].size();

    // Assert that valid conversion exists to dim4
    assert(f % (height * width) == 0);
    const size_t depth = f / (height * width);

    // Create output array
    // Todo: Add count: m when done with testing
    dim4 output = dim4(30, dim3(height, dim2(width, dim1(depth, 0.0))));

    // Loop through input array and place values into respective output position
    // TODO: Replace the i < 1 with i < m for final test
    for (size_t i = 0; i < 30; i++)
    {
	    for (size_t v = 0; v < height; v++)
	    {
		    for (size_t w = 0; w < width; w++)
		    {
			    for (size_t c = 0; c < depth; c++)
			    {
                    const size_t index = c * width * height + v * width + w;
                    output[i][v][w][c] = (static_cast<double>(input[i][index])) / 255.0;
			    }
		    }
	    }
    }
    return output;
}

void print_array(const dim2& arr)
{
    const size_t m = arr.size();
    const size_t n = arr[0].size();

    for (size_t i = 0; i < m; i++)
    {
	    for (size_t k = 0; k < n; k++)
	    {
            std::cout << arr[i][k] << ", ";
	    }
        std::cout << "\n";
    }
    std::cout << "\n";
}

double compute_cross_entropy_loss(const dim2& output, const dim2& labels)
{
    const size_t classes = output.size();
    assert(classes == labels.size());
    std::cout << "The number of classes of output is: " << output[0].size() << "\n";
    std::cout << "The number of classes of labels is: " << labels[0].size() << "\n";
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


dim2 reshape_labels_array(const std::vector<unsigned char>& labels, size_t n_classes)
{
    // Get size
    // TODO: Change m when done with initial testing procedures
    const size_t m = 30;

    // Create output matrix of size (n_classes, m)
    dim2 output = dim2(n_classes, dim1(m, 0.0));

    // Fill the output matrix
    // TODO: Change 30 to m when done with testing
    for (size_t i = 0; i < 30; i++)
    {
        output[static_cast<size_t>(labels[i])][i] = 1.0;
    }

    return output;
}


