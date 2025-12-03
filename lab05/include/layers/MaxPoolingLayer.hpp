#pragma once
#include <Layer.hpp>

class MaxPoolingLayer : public Layer {
private:
    int pool_size;
    Tensor last_input;
    Tensor max_indices; // Для запоминания, откуда был взят максимум

public:
    MaxPoolingLayer(int pool_size);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient, double learning_rate) override;
};