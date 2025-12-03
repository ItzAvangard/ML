#include <layers/MaxPoolingLayer.hpp>
#include <limits>
#include <cmath>

MaxPoolingLayer::MaxPoolingLayer(int p_size) : pool_size(p_size) {}

Tensor MaxPoolingLayer::forward(const Tensor& input) {
    this->last_input = input;

    int output_height = input.height / pool_size;
    int output_width = input.width / pool_size;
    Tensor output(input.depth, output_height, output_width);
    // Индексы храним в тензоре того же размера, что и выход
    this->max_indices = Tensor(input.depth, output_height, output_width);

    for (int d = 0; d < input.depth; ++d) {
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                double max_val = -std::numeric_limits<double>::infinity();
                int max_idx = -1;

                // Поиск максимума в окне pool_size x pool_size
                for (int py = 0; py < pool_size; ++py) {
                    for (int px = 0; px < pool_size; ++px) {
                        int current_y = y * pool_size + py;
                        int current_x = x * pool_size + px;
                        double val = input.at(d, current_y, current_x);
                        if (val > max_val) {
                            max_val = val;
                            // Сохраняем "плоский" индекс внутри окна
                            max_idx = py * pool_size + px;
                        }
                    }
                }
                output.at(d, y, x) = max_val;
                max_indices.at(d, y, x) = max_idx;
            }
        }
    }
    return output;
}

Tensor MaxPoolingLayer::backward(const Tensor& output_gradient, double learning_rate) {
    Tensor input_gradient(last_input.depth, last_input.height, last_input.width); // Инициализирован нулями

    int output_height = output_gradient.height;
    int output_width = output_gradient.width;
    
    for (int d = 0; d < output_gradient.depth; ++d) {
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                double grad = output_gradient.at(d, y, x);
                int max_idx = static_cast<int>(max_indices.at(d, y, x));
                
                // Восстанавливаем 2D-координаты из плоского индекса
                int max_py = max_idx / pool_size;
                int max_px = max_idx % pool_size;

                // Получаем абсолютные координаты в исходном тензоре
                int original_y = y * pool_size + max_py;
                int original_x = x * pool_size + max_px;

                // Передаем градиент только тому нейрону, который был максимальным
                input_gradient.at(d, original_y, original_x) = grad;
            }
        }
    }
    return input_gradient;
}