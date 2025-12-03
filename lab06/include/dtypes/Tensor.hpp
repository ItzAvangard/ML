#pragma once
#include <vector>
#include <numeric>
#include <stdexcept>

struct Tensor {
    int channels; // Новое измерение для 4D (например, количество фильтров)
    int depth;
    int height;
    int width;
    std::vector<double> data;

    // Конструктор для 3D (как раньше)
    Tensor(int d = 0, int h = 0, int w = 0) 
        : channels(1), depth(d), height(h), width(w), data(d * h * w, 0.0) {}
    
    // Новый конструктор для 4D
    Tensor(int c, int d, int h, int w)
        : channels(c), depth(d), height(h), width(w), data(c * d * h * w, 0.0) {}

    // Доступ к элементам 3D-тензора (обратная совместимость)
    double& at(int d, int h, int w) {
        return data[d * (height * width) + h * width + w];
    }
    const double& at(int d, int h, int w) const {
        return data[d * (height * width) + h * width + w];
    }
    
    // Доступ к элементам 4D-тензора
    double& at(int c, int d, int h, int w) {
        return data[c * (depth * height * width) + d * (height * width) + h * width + w];
    }
    const double& at(int c, int d, int h, int w) const {
        return data[c * (depth * height * width) + d * (height * width) + h * width + w];
    }
    
    // Остальные методы остаются без изменений...
    static Tensor from_vector(const std::vector<double>& vec, int d, int h, int w) {
        if (vec.size() != d * h * w) {
            throw std::invalid_argument("Vector size does not match tensor dimensions");
        }
        Tensor t(d, h, w);
        t.data = vec;
        return t;
    }
};