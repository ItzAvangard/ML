#pragma once
// Оставляем #define здесь, раз он уже есть, но в main.cpp его нужно убрать
#define STB_IMAGE_IMPLEMENTATION
#include <ext/stb_image.h>
#include <vector>
#include <string>
#include <stdexcept>

std::vector<double> load_image(const std::string& path) {
    int width, height, channels;
    unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 1); // 1 for grayscale
    if (img == nullptr) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    if (width != 28 || height != 28) {
        stbi_image_free(img);
        throw std::runtime_error("Image must be 28x28 pixels.");
    }

    std::vector<double> image_data(width * height);
    for (int i = 0; i < width * height; ++i) {
        // Нормализуем 
        image_data[i] = static_cast<double>(img[i]) / 255.0;
    }
    stbi_image_free(img);
    return image_data;
}