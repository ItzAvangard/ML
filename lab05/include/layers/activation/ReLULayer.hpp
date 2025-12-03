#pragma once
#include <Layer.hpp>

class ReLULayer : public Layer {
private:
    Tensor last_input;

public:
    ReLULayer() = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient, double learning_rate) override;
};