#ifndef MNISTLOADER_H
#define MNISTLOADER_H

#include <vector>
#include <string>

namespace MNISTLoader {
    extern const int IMAGE_WIDTH;
    extern const int IMAGE_HEIGHT;
    extern const int IMAGE_SIZE;
    extern const int NUM_IMAGES;
    extern const int TENTH_NUM_IMAGES;

    int read_int(std::ifstream& file);
    void load_mnist(const std::string& image_file, const std::string& label_file,
        std::vector<std::vector<unsigned char>>& images, std::vector<unsigned char>& labels);
}

#endif // MNISTLOADER_H
