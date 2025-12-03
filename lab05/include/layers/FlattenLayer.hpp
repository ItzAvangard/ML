#pragma once
#include <Layer.hpp>

class FlattenLayer : public Layer {
private:
    int last_input_depth, last_input_height, last_input_width;

public:
    FlattenLayer() = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient, double learning_rate) override;
};