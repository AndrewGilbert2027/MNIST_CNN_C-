#include "MNISTLoader.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace MNISTLoader {
    const int IMAGE_WIDTH = 28;
    const int IMAGE_HEIGHT = 28;
    const int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
    const int NUM_IMAGES = 60000;
    const int TENTH_NUM_IMAGES = NUM_IMAGES / 10;

    int read_int(std::ifstream& file) {
        unsigned char buffer[4];
        file.read(reinterpret_cast<char*>(buffer), 4);
        return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
    }

    void load_mnist(const std::string& image_file, const std::string& label_file,
        std::vector<std::vector<unsigned char>>& images, std::vector<unsigned char>& labels) {

        std::ifstream image_stream(image_file, std::ios::binary);
        std::ifstream label_stream(label_file, std::ios::binary);

        if (!image_stream.is_open() || !label_stream.is_open()) {
            throw std::runtime_error("Unable to open MNIST data files.");
        }

        int magic_number = read_int(image_stream);
        const int num_images = read_int(image_stream);
        const int num_rows = read_int(image_stream);
        const int num_cols = read_int(image_stream);

        int label_magic_number = read_int(label_stream);
        const int num_labels = read_int(label_stream);

        if (num_images != num_labels || num_rows != IMAGE_HEIGHT || num_cols != IMAGE_WIDTH) {
            throw std::runtime_error("MNIST data file header mismatch.");
        }

        // Only load in 10% of the training data for testing and performance issues
        images.resize(TENTH_NUM_IMAGES, std::vector<unsigned char>(IMAGE_SIZE));
        labels.resize(TENTH_NUM_IMAGES);

        for (int i = 0; i < TENTH_NUM_IMAGES; ++i) {
            image_stream.read(reinterpret_cast<char*>(images[i].data()), IMAGE_SIZE);
            label_stream.read(reinterpret_cast<char*>(&labels[i]), 1);
        }
    }
}
