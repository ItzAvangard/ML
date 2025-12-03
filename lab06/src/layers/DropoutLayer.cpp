#include <layers/DropoutLayer.hpp>
#include <random>
#include <stdexcept>

DropoutLayer::DropoutLayer(double rate) : dropout_rate(rate) {
    if (rate < 0.0 || rate >= 1.0) {
        throw std::invalid_argument("Dropout rate must be in [0, 1)");
    }
}

void DropoutLayer::set_training_mode(bool training_mode) {
    this->is_training = training_mode;
}

Tensor DropoutLayer::forward(const Tensor& input) {
    // В режиме оценки (вывода) слой ничего не делает, просто пропускает данные
    if (!is_training) {
        return input;
    }

    // В режиме обучения применяем dropout
    Tensor output = input; // Копируем структуру и данные
    this->dropout_mask = Tensor(input.depth, input.height, input.width);
    
    // Генератор случайных чисел
    static std::mt19937 gen(std::random_device{}());
    // Распределение, которое с вероятностью `dropout_rate` вернет `true` (т.е. нейрон будет отключен)
    std::bernoulli_distribution dis(dropout_rate);

    double scale_factor = 1.0 / (1.0 - dropout_rate); // Масштаб для Inverted Dropout

    for (size_t i = 0; i < input.data.size(); ++i) {
        if (dis(gen)) { // Если true, отключаем нейрон
            output.data[i] = 0.0;
            dropout_mask.data[i] = 0.0; // Запоминаем, что он был отключен
        } else { // Если false, нейрон "выживает"
            output.data[i] *= scale_factor; // Масштабируем его
            dropout_mask.data[i] = 1.0; // Запоминаем, что он выжил
        }
    }
    
    return output;
}

Tensor DropoutLayer::backward(const Tensor& output_gradient, double learning_rate) {
    // В режиме оценки слой не менял данные, поэтому просто пропускает градиент
    if (!is_training) {
        return output_gradient;
    }

    // В режиме обучения градиент проходит только через "выжившие" нейроны
    Tensor input_gradient = output_gradient; // Копируем
    double scale_factor = 1.0 / (1.0 - dropout_rate);

    for (size_t i = 0; i < output_gradient.data.size(); ++i) {
        if (dropout_mask.data[i] == 0.0) { // Если нейрон был отключен
            input_gradient.data[i] = 0.0;
        } else { // Если нейрон выжил
            input_gradient.data[i] *= scale_factor; // Применяем то же масштабирование
        }
    }

    return input_gradient;
}