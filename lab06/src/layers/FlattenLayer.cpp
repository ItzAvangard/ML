#include <layers/FlattenLayer.hpp>
#include <dtypes/Tensor.hpp> 


Tensor FlattenLayer::forward(const Tensor& input) {
    // 1. Сохраняем размеры входа для использования в обратном проходе.
    this->last_input_depth = input.depth;
    this->last_input_height = input.height;
    this->last_input_width = input.width;

    // 2. Вычисляем общий размер "сплющенного" вектора.
    int flattened_size = input.depth * input.height * input.width;

    // 3. Создаем выходной тензор с новыми размерами.
    Tensor output(flattened_size, 1, 1);

    // 4. Копируем данные. Поскольку std::vector уже хранит данные линейно в памяти,
    // это простая операция присваивания векторов.
    output.data = input.data;

    return output;
}


Tensor FlattenLayer::backward(const Tensor& output_gradient, double learning_rate) {
    // Параметр learning_rate здесь не нужен, но должен присутствовать
    // для соответствия интерфейсу базового класса Layer.
    // (void)learning_rate; // Это можно использовать для подавления предупреждений компилятора.

    // 1. Создаем тензор для градиента по входу, используя сохраненные размеры.
    Tensor input_gradient(last_input_depth, last_input_height, last_input_width);

    // 2. Копируем данные градиента. Метаданные тензора (его размеры d, h, w)
    // обеспечат правильную интерпретацию этих данных предыдущим слоем
    // (например, MaxPoolingLayer).
    input_gradient.data = output_gradient.data;

    return input_gradient;
}