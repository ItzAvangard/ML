#include <layers/activation/SigmoidLayer.hpp>
#include <cmath> // Для std::exp

Tensor SigmoidLayer::forward(const Tensor& input) {
    Tensor output(input.depth, input.height, input.width);

    for (size_t i = 0; i < input.data.size(); ++i) {
        // Формула сигмоиды: 1 / (1 + e^(-x))
        output.data[i] = 1.0 / (1.0 + std::exp(-input.data[i]));
    }

    // Сохраняем результат для использования в обратном проходе
    this->last_output = output;
    
    return output;
}

// Обратный проход: распространение градиента через слой
Tensor SigmoidLayer::backward(const Tensor& output_gradient, double learning_rate) {

    // Создаем тензор для градиента по входу с теми же размерами
    Tensor input_gradient(output_gradient.depth, output_gradient.height, output_gradient.width);

    for (size_t i = 0; i < output_gradient.data.size(); ++i) {
        // Производная сигмоиды: y * (1 - y), где y - это выход (last_output)
        double sigmoid_output = this->last_output.data[i];
        double sigmoid_derivative = sigmoid_output * (1.0 - sigmoid_output);

        // Применяем цепное правило: dL/dX = dL/dY * dY/dX
        // dL/dY - это входящий output_gradient
        // dY/dX - это производная сигмоиды
        input_gradient.data[i] = output_gradient.data[i] * sigmoid_derivative;
    }

    return input_gradient;
}