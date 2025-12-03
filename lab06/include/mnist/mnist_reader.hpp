#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>

// Function to reverse integer for MNIST file format
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Function to read MNIST images
std::vector<std::vector<double>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);

        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);

        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);

        std::vector<std::vector<double>> images(number_of_images);
        for (int i = 0; i < number_of_images; ++i) {
            images[i].resize(n_rows * n_cols);
            for (int r = 0; r < n_rows; ++r) {
                for (int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    images[i][r * n_cols + c] = (double)temp / 255.0;
                }
            }
        }
        return images;
    }
    else {
        throw std::runtime_error("Cannot open file `" + path + "`!");
    }
}

// Function to read MNIST labels
std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_labels = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);
        std::vector<int> labels(number_of_labels);
        for (int i = 0; i < number_of_labels; ++i) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            labels[i] = (int)temp;
        }
        return labels;
    }
    else {
        throw std::runtime_error("Cannot open file `" + path + "`!");
    }
}