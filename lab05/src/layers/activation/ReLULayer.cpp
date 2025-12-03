#include <layers/activation/ReLULayer.hpp>
#include <dtypes/Tensor.hpp>
#include <algorithm> // Для std::max


Tensor ReLULayer::forward(const Tensor& input) {
    // 1. Кэшируем входной тензор. Это необходимо для обратного прохода,
    //    так как производная ReLU зависит от знака входа.
    this->last_input = input;

    // 2. Создаем выходной тензор с теми же размерами, что и входной.
    Tensor output(input.depth, input.height, input.width);
    output.channels = input.channels; // Сохраняем и 4-е измерение, если оно есть

    // 3. Применяем функцию ReLU к каждому элементу.
    for (size_t i = 0; i < input.data.size(); ++i) {
        // f(x) = max(0, x)
        output.data[i] = std::max(0.0, input.data[i]);
    }

    return output;
}


Tensor ReLULayer::backward(const Tensor& output_gradient, double learning_rate) {
    // (void)learning_rate; // Подавляем предупреждение компилятора "unused parameter"

    // 1. Создаем тензор для градиента по входу с теми же размерами.
    Tensor input_gradient(last_input.depth, last_input.height, last_input.width);
    input_gradient.channels = last_input.channels;

    // 2. Вычисляем градиент, используя цепное правило: dL/dX = dL/dY * dY/dX
    for (size_t i = 0; i < output_gradient.data.size(); ++i) {
        // Производная ReLU (dY/dX) равна 1, если вход (X) был > 0, и 0 в противном случае.
        if (this->last_input.data[i] > 0) {
            // Если вход был положительным, градиент проходит без изменений.
            input_gradient.data[i] = output_gradient.data[i];
        } else {
            // Если вход был отрицательным или нулевым, градиент блокируется (становится 0).
            input_gradient.data[i] = 0.0;
        }
    }

    return input_gradient;
}