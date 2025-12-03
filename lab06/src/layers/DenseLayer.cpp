#include <layers/DenseLayer.hpp>
#include <stdexcept>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>

// Конструктор слоя
DenseLayer::DenseLayer(int in_size, int out_size) 
    : input_size(in_size), output_size(out_size) {
        
    // Инициализируем тензоры для весов и смещений
    weights = Tensor(output_size, input_size, 1);
    biases = Tensor(output_size, 1, 1);
    
    // ИИ предлолагает использовать инициализацию Xavier
    double limit = std::sqrt(6.0 / (input_size + output_size));

    // Используем современный генератор случайных чисел C++
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-limit, limit);

    for (double& weight : weights.data) {
        weight = dis(gen);
    }
    
    // Смещения (biases) обычно инициализируются нулями
    for (double& bias : biases.data) {
        bias = 0.0;
    }
}

// Прямой проход: output = weights * input + biases
Tensor DenseLayer::forward(const Tensor& input) {
    // Проверка соответствия размеров
    if (input.data.size() != input_size) {
        throw std::invalid_argument("DenseLayer forward: input size mismatch.");
    }

    // Сохраняем вход для использования в обратном проходе
    last_input = input;

    // Создаем выходной тензор
    Tensor output(output_size, 1, 1);

    // Выполняем матрично-векторное умножение и добавляем смещение
    for (int i = 0; i < output_size; ++i) {
        double sum = 0.0;
        for (int j = 0; j < input_size; ++j) {
            // weights.at(i, j, 0) - это эквивалент W_ij
            sum += weights.at(i, j, 0) * input.data[j];
        }
        output.data[i] = sum + biases.data[i];
    }

    return output;
}

// Обратный проход
Tensor DenseLayer::backward(const Tensor& output_gradient, double learning_rate) {
    // Проверка соответствия размеров градиента
    if (output_gradient.data.size() != output_size) {
        throw std::invalid_argument("DenseLayer backward: gradient size mismatch.");
    }

    // 1. Вычисляем градиент для передачи на предыдущий слой (dL/dX)
    // dL/dX = W^T * dL/dY, где dL/dY - это output_gradient
    Tensor input_gradient(input_size, 1, 1);
    for (int j = 0; j < input_size; ++j) {
        double sum = 0.0;
        for (int i = 0; i < output_size; ++i) {
            sum += weights.at(i, j, 0) * output_gradient.data[i];
        }
        input_gradient.data[j] = sum;
    }

    // 2. Вычисляем градиенты для весов (dL/dW) и смещений (dL/db) и обновляем их
    // dL/dW_ij = dL/dY_i * X_j (внешнее произведение градиента выхода на вход)
    // dL/db_i = dL/dY_i
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            double weight_gradient = output_gradient.data[i] * last_input.data[j];
            // Обновляем вес (шаг градиентного спуска)
            weights.at(i, j, 0) -= learning_rate * weight_gradient;
        }
        // Обновляем смещение
        biases.data[i] -= learning_rate * output_gradient.data[i];
    }
    
    // Возвращаем градиент по входу для следующего слоя
    return input_gradient;
}

// Сохранение весов и смещений в бинарный файл
void DenseLayer::save_weights(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(weights.data.data()), weights.data.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(biases.data.data()), biases.data.size() * sizeof(double));
}

// Загрузка весов и смещений из бинарного файла
void DenseLayer::load_weights(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(weights.data.data()), weights.data.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(biases.data.data()), biases.data.size() * sizeof(double));
     if (!file) {
        throw std::runtime_error("Error reading weights for DenseLayer. File might be corrupted or too short.");
    }
}