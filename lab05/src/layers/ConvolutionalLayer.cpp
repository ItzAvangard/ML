// ConvolutionalLayer.cpp
#include <layers/ConvolutionalLayer.hpp>
#include <stdexcept>
#include <random>
#include <cmath>
#include <fstream>

ConvolutionalLayer::ConvolutionalLayer(int input_depth, int n_filters, int k_size)
    : num_filters(n_filters), kernel_size(k_size) {

    // Инициализация тензоров для ядер (фильтров) и смещений
    kernels = Tensor(num_filters, input_depth, kernel_size, kernel_size);
    biases = Tensor(num_filters, 1, 1);

    // --- Оптимизация: Инициализация весов по методу He ---
    // Лучше подходит для ReLU, чем Ксавье. Распределение с stddev = sqrt(2 / fan_in)
    // fan_in для сверточного слоя = input_depth * kernel_size * kernel_size
    double fan_in = input_depth * kernel_size * kernel_size;
    double stddev = std::sqrt(2.0 / fan_in);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, stddev);

    for (double& k : kernels.data) {
        k = dis(gen);
    }
    // Смещения инициализируем нулями
    for (double& b : biases.data) {
        b = 0.0;
    }
}

Tensor ConvolutionalLayer::forward(const Tensor& input) {
    this->last_input = input;
    
    int output_height = input.height - kernel_size + 1;
    int output_width = input.width - kernel_size + 1;
    Tensor output(num_filters, output_height, output_width);

    // Проход по каждому фильтру
    for (int f = 0; f < num_filters; ++f) {
        // "Скользим" окном по входу
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                double sum = 0.0;
                // Выполняем свертку (по сути, dot product)
                for (int d = 0; d < input.depth; ++d) {
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            sum += input.at(d, y + ky, x + kx) * kernels.at(f, d, ky, kx);
                        }
                    }
                }
                output.at(f, y, x) = sum + biases.at(f, 0, 0);
            }
        }
    }
    return output;
}

Tensor ConvolutionalLayer::backward(const Tensor& output_gradient, double learning_rate) {
    Tensor kernel_gradient(kernels.channels, kernels.depth, kernels.height, kernels.width);
    Tensor bias_gradient(biases.depth, biases.height, biases.width); // Здесь depth=num_filters
    Tensor input_gradient(last_input.depth, last_input.height, last_input.width);

    int output_height = output_gradient.height;
    int output_width = output_gradient.width;

    // Вычисление градиентов
    for (int f = 0; f < num_filters; ++f) {
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                double grad = output_gradient.at(f, y, x);
                // Градиент для смещений
                bias_gradient.at(f, 0, 0) += grad;
                
                for (int d = 0; d < last_input.depth; ++d) {
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            // Градиент для ядер (фильтров)
                            kernel_gradient.at(f, d, ky, kx) += last_input.at(d, y + ky, x + kx) * grad;
                            // Градиент для входа (передается на предыдущий слой)
                            input_gradient.at(d, y + ky, x + kx) += kernels.at(f, d, ky, kx) * grad;
                        }
                    }
                }
            }
        }
    }

    // Обновление весов и смещений
    for (size_t i = 0; i < kernels.data.size(); ++i) {
        kernels.data[i] -= learning_rate * kernel_gradient.data[i];
    }
    for (size_t i = 0; i < biases.data.size(); ++i) {
        biases.data[i] -= learning_rate * bias_gradient.data[i];
    }

    return input_gradient;
}

void ConvolutionalLayer::save_weights(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(kernels.data.data()), kernels.data.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(biases.data.data()), biases.data.size() * sizeof(double));
}

void ConvolutionalLayer::load_weights(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(kernels.data.data()), kernels.data.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(biases.data.data()), biases.data.size() * sizeof(double));
    if (!file) throw std::runtime_error("Error reading weights for ConvolutionalLayer.");
}